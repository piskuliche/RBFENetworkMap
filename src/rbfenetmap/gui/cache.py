"""Memoized mapping, so a knob can be moved without re-running the MCS searches.

The measurement this module is built on: over the shipped Tyk2 set -- sixteen ligands,
a hundred and twenty pairs, eight jobs -- a full ``rbfenet plan`` takes about 2.1 s, and it
takes about 2.1 s for *every one* of the twenty-one variants in the published matrix,
whatever the planner or the selection knobs. The same run with ``--cbfe all``, which skips
mapping entirely, takes 0.5 s. Mapping is the cost, and mapping is the one stage that does
not care which planner runs afterwards.

So the GUI wraps the mapper rather than reaching into the pipeline.
:func:`~rbfenetmap.core.pipeline.build_network` already accepts a mapper *instance*, which
makes :class:`CachingMapper` a plugin like any other and needs no change to core. Moving a
selection knob then re-runs the repair and the scorer -- pure Python, and cheap -- while the
``FindMCS`` calls come back from a dict.

The cache is keyed on molblocks rather than on ligand names, for the reason
:mod:`rbfenetmap.io.networkio` gives for embedding molblocks instead of file paths: an
:class:`~rbfenetmap.core.models.AtomMapping` is indices into a particular atom ordering, and
is meaningless against a molecule that has been re-read into a different one. A name is not
an identity; the atom block is.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Callable, ClassVar

from rdkit import Chem

from rbfenetmap.core.exceptions import MappingError
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.models import AtomMapping, Ligand
from rbfenetmap.core.options import MappingOptions

logger = logging.getLogger(__name__)

__all__ = ("CachingMapper", "MappingCache", "RunCancelled")

#: Bumped when the stored shape changes, so an old file is discarded rather than
#: misread. There is nothing to migrate: every entry can be recomputed from the ligands.
_CACHE_VERSION = 1


class RunCancelled(Exception):
    """Raised inside a mapper to abandon a planning run the user has cancelled.

    Deliberately **not** a :class:`~rbfenetmap.core.exceptions.MappingError`.
    :func:`~rbfenetmap.core.pipeline.build_candidate` catches that one and turns it into a
    ``mapper_failed`` rejection, so a cancellation spelled that way would not stop the run
    at all -- it would quietly produce a network in which every pair not yet reached looks
    infeasible, which is far worse than not stopping.

    Notes
    -----
    Cancellation takes effect within one ``--mcs-timeout``. The pool that maps pairs in
    parallel waits for its in-flight searches on the way out, and an ``FindMCS`` call
    already inside RDKit cannot be interrupted from Python. What this stops is every pair
    that has not started yet, which on a large set is nearly all of them.
    """


def _molblock(ligand: Ligand) -> str:
    """Return the atom block that identifies *ligand* for caching purposes."""
    return Chem.MolToMolBlock(ligand.mol, kekulize=False)


def _key(source: Ligand, target: Ligand, mapper_name: str, options: MappingOptions) -> str:
    """Return the cache key for one directed pair under one mapper configuration.

    Directed on purpose. ``map_pair(a, b)`` and ``map_pair(b, a)`` return mappings whose
    ``cc1`` indexes a different molecule, so treating them as one entry would hand back
    correspondences pointing at the wrong ligand.
    """
    payload = json.dumps(
        {
            "version": _CACHE_VERSION,
            "mapper": mapper_name,
            # asdict rather than the field list: it recurses into core_pruning, so a
            # pruning policy change invalidates the entry it should invalidate.
            "options": dataclasses.asdict(options),
            "source": _molblock(source),
            "target": _molblock(target),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _encode(result: AtomMapping | MappingError) -> dict:
    """Serialize a cache entry, success or failure."""
    if isinstance(result, MappingError):
        return {"error": str(result)}
    return {
        "cc1": list(result.cc1),
        "cc2": list(result.cc2),
        "sc1": list(result.sc1),
        "sc2": list(result.sc2),
        "n_atoms_1": result.n_atoms_1,
        "n_atoms_2": result.n_atoms_2,
        "method": result.method,
    }


def _decode(entry: dict) -> AtomMapping | MappingError:
    """Rebuild a cache entry.

    The mapping goes back through :class:`~rbfenetmap.core.models.AtomMapping`'s own
    constructor, so a truncated or hand-edited file fails its validation here rather than
    surfacing as a malformed correspondence much later.
    """
    if "error" in entry:
        return MappingError(entry["error"])
    return AtomMapping(
        cc1=tuple(entry["cc1"]),
        cc2=tuple(entry["cc2"]),
        sc1=tuple(entry["sc1"]),
        sc2=tuple(entry["sc2"]),
        n_atoms_1=entry["n_atoms_1"],
        n_atoms_2=entry["n_atoms_2"],
        method=entry["method"],
    )


class MappingCache:
    """Atom mappings kept by pair, mapper and mapping options.

    Parameters
    ----------
    path : Path, optional
        JSON file to load on construction and write on :meth:`save`. ``None`` keeps the
        cache in memory for the life of the process.

    Attributes
    ----------
    hits, misses : int
        Lookup counters, so the GUI can say why a run was fast.

    Notes
    -----
    Thread-safe. :func:`~rbfenetmap.core.pipeline.evaluate_pairs` maps pairs across a
    thread pool, so several lookups and stores are genuinely concurrent.

    **A failed mapping is cached too.** A pair no MCS search can relate is precisely the
    pair that costs the full ``--mcs-timeout`` to fail, every time, and precisely the one
    whose answer will not change when a planner knob moves.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.hits = 0
        self.misses = 0
        self._entries: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path is not None and path.exists():
            self._load()

    def _load(self) -> None:
        """Read the cache file, discarding it if it is unreadable or of another version.

        A corrupt cache is a performance problem, never a correctness one: everything in
        it can be recomputed. So this warns and starts empty rather than raising and
        leaving the user with a tool that will not open.
        """
        assert self.path is not None
        try:
            data = json.loads(self.path.read_text())
            if data.get("version") != _CACHE_VERSION:
                logger.info("Ignoring mapping cache %s: written by another version", self.path)
                return
            self._entries = dict(data["entries"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unreadable mapping cache %s: %s", self.path, exc)

    def save(self) -> None:
        """Write the cache to :attr:`path`, atomically. A no-op with no path set.

        Written through a temporary file in the same directory and then renamed, so an
        interrupted save leaves the previous cache intact instead of a half-written file
        that the next load would discard.
        """
        if self.path is None:
            return
        with self._lock:
            payload = {"version": _CACHE_VERSION, "entries": dict(self._entries)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload))
        os.replace(temporary, self.path)

    def get(self, key: str) -> AtomMapping | MappingError | None:
        """Return the cached result for *key*, or ``None`` on a miss."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
        return _decode(entry)

    def put(self, key: str, result: AtomMapping | MappingError) -> None:
        """Store *result* under *key*."""
        encoded = _encode(result)
        with self._lock:
            self._entries[key] = encoded

    def clear(self) -> None:
        """Drop every entry and reset the counters."""
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class CachingMapper(AbstractMapper):
    """A mapper that remembers what it has already mapped, and can be cancelled.

    Parameters
    ----------
    wrapped : AbstractMapper
        The real mapper. Every miss is delegated to it verbatim.
    cache : MappingCache, optional
        Shared store. A fresh in-memory one is made if omitted.
    should_cancel : callable, optional
        Polled before each pair. Returning ``True`` raises :class:`RunCancelled`.

    Notes
    -----
    :attr:`name` is set on the instance to the wrapped mapper's, shadowing the class
    attribute. That is what keeps a cached run indistinguishable from an uncached one:
    the name is recorded on every :class:`~rbfenetmap.core.models.AtomMapping` as its
    ``method`` and is serialized into the network JSON, so a ``CachingMapper`` that
    reported its own name would make every planned network say it was mapped by something
    that is not a mapping algorithm at all.
    """

    name: ClassVar[str] = "caching"

    def __init__(
        self,
        wrapped: AbstractMapper,
        cache: MappingCache | None = None,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.wrapped = wrapped
        self.cache = cache if cache is not None else MappingCache()
        self.should_cancel = should_cancel
        # Instance attribute, shadowing the ClassVar. See the class notes.
        self.name = wrapped.name

    def supports_pair(self, source: Ligand, target: Ligand) -> bool:
        """Delegate the cheap pre-check; it is not worth caching."""
        return self.wrapped.supports_pair(source, target)

    def map_pair(self, source: Ligand, target: Ligand, options: MappingOptions) -> AtomMapping:
        """Return the correspondence, from the cache when it is there.

        Raises
        ------
        RunCancelled
            If *should_cancel* returns true.
        rbfenetmap.core.exceptions.MappingError
            As the wrapped mapper would, whether the failure is fresh or remembered.
        """
        if self.should_cancel is not None and self.should_cancel():
            raise RunCancelled(f"Cancelled before mapping {source.name}~{target.name}.")

        key = _key(source, target, self.wrapped.name, options)
        cached = self.cache.get(key)
        if cached is not None:
            if isinstance(cached, MappingError):
                raise cached
            return cached

        try:
            mapping = self.wrapped.map_pair(source, target, options)
        except MappingError as exc:
            self.cache.put(key, exc)
            raise
        self.cache.put(key, mapping)
        return mapping
