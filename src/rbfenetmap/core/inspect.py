"""Everything known about one edge, gathered once so two front-ends cannot disagree.

``rbfenet inspect`` and the GUI's soft-core panel ask the same question of an edge, and a
question answered in two places is answered differently the moment either place gains a
field. So the resolution rule and the fact set live here, and each caller only formats.

Deliberately free of Amber masks and depictions. This module is in ``core``, which is the
bottom layer and imports nothing from :mod:`rbfenetmap.io` or :mod:`rbfenetmap.viz`; masks
belong to the exporter's world and pictures to the renderer's, so the *caller* adds them --
which is what :func:`rbfenetmap.cli.commands.cmd_inspect` already did.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal, Mapping, Sequence

from rbfenetmap.core.models import EdgeKind, Network, Transformation, parse_edge_key

__all__ = ("SEARCH_SCOPES", "SearchScope", "edge_facts", "resolve_edge")

SearchScope = Literal["edges", "candidates", "any"]

#: Where :func:`resolve_edge` may look, and what each choice means.
#:
#: The scope is a parameter rather than a fixed rule because **one key can name two
#: different edges**. Under ``cbfe_mode="bridge"`` a pair the mapper could not relate can be
#: both a selected counterpoised edge and a rejected relative candidate, with the same
#: ``source~target`` key. Searching a fixed order silently answers only one of the two
#: questions a caller might be asking.
SEARCH_SCOPES: tuple[SearchScope, ...] = ("edges", "candidates", "any")


def _finite(value: float | None) -> float | None:
    """Return *value*, or ``None`` when it is not a finite number.

    ``EdgeScore.total`` is ``math.inf`` on every infeasible candidate, and ``json.dumps``
    renders that as the bare token ``Infinity``, which ``JSON.parse`` rejects outright. A
    browser client fetching a rejected edge would fail with a console error and nothing to
    show for it on the server.

    :mod:`rbfenetmap.io.networkio` already spells the same fix as ``total if feasible else
    None``; this is that rule applied to every number leaving this module, so a descriptor
    that goes non-finite in future cannot reintroduce the bug.
    """
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _finite_map(values: Mapping[str, float]) -> dict[str, float | None]:
    """Apply :func:`_finite` across a mapping of numbers."""
    return {key: _finite(value) for key, value in values.items()}


def _search(candidates: Iterable[Transformation], key: str) -> Transformation | None:
    """Find *key* among *candidates*, tolerating the direction it was stored in.

    Three passes, narrowing: the exact directed key, then the reverse, then either
    orientation. The order matters because a caller who typed ``a~b`` meant ``a~b`` if one
    exists, and because ``candidates`` may legitimately hold both directions of a pair --
    ``pair_strategy="all_pairs"`` enumerates permutations rather than combinations.

    The reverse pass is not a nicety. Every planner orients its selected edges through
    :func:`~rbfenetmap.core.models.orient_edge`, so a selected edge's key routinely differs
    from the key of the candidate it came from: on a six-ligand aptamer network, three of
    six selected edges have keys that appear nowhere in ``candidates``.
    """
    pool = list(candidates)
    source, target = parse_edge_key(key)
    reversed_key = f"{target}~{source}"
    wanted = tuple(sorted((source, target)))

    for match in (lambda e: e.key == key, lambda e: e.key == reversed_key):
        found = next((edge for edge in pool if match(edge)), None)
        if found is not None:
            return found
    return next((edge for edge in pool if edge.unordered_key == wanted), None)


def resolve_edge(network: Network, key: str, *, scope: SearchScope = "any") -> Transformation:
    """Return the transformation *key* names, searching *scope*.

    Parameters
    ----------
    network : Network
    key : str
        An ``"a~b"`` edge key, in either direction.
    scope : {"edges", "candidates", "any"}
        ``"edges"`` searches the selected network only, ``"candidates"`` the scored pool
        only, ``"any"`` the selected network and then the pool.

        Choose deliberately. ``"any"`` answers "tell me about this pair" and will return the
        selected edge whenever there is one; ``"candidates"`` is the only way to reach the
        relative candidate for a pair that was ultimately bridged with a counterpoised edge.

        ``"edges"`` is also the only scope that is correct after ``--consistency graph``,
        which rewrites the selected edges' mappings and leaves the candidate pool holding
        the superseded ones.

    Returns
    -------
    Transformation

    Raises
    ------
    ValueError
        If *key* is malformed, or names no edge in *scope*. The message lists what is
        available, because a mistyped ligand name is the usual cause.
    """
    if scope not in SEARCH_SCOPES:
        raise ValueError(f"scope must be one of {list(SEARCH_SCOPES)}; got {scope!r}.")
    parse_edge_key(key)  # Raises with its own message on a malformed key.

    pools: dict[SearchScope, Sequence[Transformation]] = {
        "edges": network.edges,
        "candidates": network.candidates,
        "any": (*network.edges, *network.candidates),
    }
    found = _search(pools[scope], key)
    if found is not None:
        return found

    known = sorted({edge.key for edge in pools[scope]})
    where = "the selected network" if scope == "edges" else "the candidate pool" if scope == "candidates" else "network"
    raise ValueError(f"Edge {key!r} is not in {where}. Known: {known}.")


def edge_facts(network: Network, edge: Transformation) -> dict[str, Any]:
    """Return everything known about *edge*, as JSON-ready primitives.

    Parameters
    ----------
    network : Network
        Supplies whether the pair was selected, and which endpoints were invented.
    edge : Transformation
        Usually from :func:`resolve_edge`.

    Returns
    -------
    dict
        ``key``, ``source``, ``target``, ``kind``, ``selected``, ``feasible``, ``mapper``,
        the atom counts, ``cost`` (``None`` when infeasible -- see :func:`_finite`),
        ``rejections``, the repair's ``regions_before`` / ``regions_after`` / ``repaired`` /
        ``repair_rejection`` / ``n_demoted`` / ``trace``, ``descriptors``, ``contributions``, and ``synthetic``:
        the endpoints this package invented rather than read from a file.

    Notes
    -----
    ``selected`` is a statement about the **pair**, not this object: a pair bridged by a
    counterpoised edge is selected even when the transformation in hand is the relative
    candidate that was refused. That is the sense ``rbfenet inspect`` has always printed.

    ``trace`` is returned whole and in order. :meth:`Transformation.reversed` prepends a
    note recording that the sides below refer to the other orientation, so a caller that
    drops or reflows the first line makes the rest of the trace disagree with the pictures
    beside it.
    """
    selected = any(candidate.unordered_key == edge.unordered_key for candidate in network.edges)
    synthetic = [
        name
        for name in (edge.source, edge.target)
        if (ligand := network.ligands.get(name)) is not None and ligand.synthetic
    ]
    return {
        "key": edge.key,
        "source": edge.source,
        "target": edge.target,
        "kind": edge.kind.value,
        "selected": selected,
        "feasible": edge.feasible,
        "mapper": edge.mapping.method,
        "n_common_core": edge.mapping.n_common_core,
        "n_softcore_1": edge.mapping.n_softcore_1,
        "n_softcore_2": edge.mapping.n_softcore_2,
        "n_atoms_1": edge.mapping.n_atoms_1,
        "n_atoms_2": edge.mapping.n_atoms_2,
        "cost": _finite(edge.score.total) if edge.feasible else None,
        "scorer": edge.score.scorer,
        "rejections": [reason.value for reason in edge.score.rejections],
        "repaired": edge.repair.applied,
        # Which reason, if any, the *repair* itself raised -- as distinct from one the
        # precheck or the geometry gate raised afterwards. The distinction matters to a
        # caller drawing the partition: when the repair gives up, the mapping stored on the
        # edge is where it stopped, not a final answer, so "common core 12" can sit beside
        # a `no_common_core` rejection and read as a contradiction unless it is explained.
        "repair_rejection": edge.repair.rejection.value if edge.repair.rejection is not None else None,
        "regions_before": edge.repair.n_fragments_before,
        "regions_after": edge.repair.n_fragments_after,
        "n_demoted": edge.repair.n_demoted,
        "trace": list(edge.repair.trace),
        "descriptors": _finite_map(edge.score.descriptors),
        "contributions": _finite_map(edge.score.contributions),
        "synthetic": synthetic,
        "counterpoised": edge.kind is EdgeKind.CBFE,
    }
