"""Partition ligands into clusters that share a common core.

The question this module answers is not "which two ligands are similar?" -- the mapper and
the fingerprint prefilter already answer that -- but **"does this whole group share a core
worth building a sub-network around?"** Those are different questions, and the second is
the one a clustered network needs: a group of ligands all sharing one scaffold can be
cycled internally with small, trustworthy soft-cores, and joined to the rest of the series
by a single bridging edge.

The answer comes from the *N-way* MCS of the group, via
:func:`~rbfenetmap.core.mcs.mcs_query_many`. Intersecting pairwise MCS results instead
would not merely be slower, it would be meaningless: each pairwise substructure is its own
query molecule with its own atom indexing, so there is nothing to intersect. On a
three-scaffold benzamide / cyclohexanecarboxamide / thiophene-2-carboxamide set the N-way
signal is sharp -- within-scaffold cores of 9, 9, and 8 heavy atoms against 4 for the full
set, on ligands of 8 to 10 heavy atoms each -- which is what makes a single threshold a
workable control.

Why agglomerative
-----------------
The merge is seeded by the **pairwise-feasible RBFE graph** and only ever considers cluster
pairs adjacent in it. That bound matters twice over. It keeps the number of N-way searches
near-linear in the number of merges rather than quadratic in the ligand count, and it means
a cluster can never contain two ligands with no feasible mapping between them -- which
would be a cluster whose internal network cannot be built.

Merging the *best available* pair each round, rather than accepting the first admissible
one, is what makes the result independent of iteration order. Ties break on the union's
total cost and then on the sorted member names, so the partition is reproducible.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from rbfenetmap.core.mcs import mcs_query_many

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

    from rbfenetmap.core.models import Ligand
    from rbfenetmap.core.options import ClusteringPolicy, MappingOptions

__all__ = ("core_clusters", "cluster_core_size", "describe_clusters")

logger = logging.getLogger(__name__)


def cluster_core_size(
    names: Iterable[str],
    ligands: "Mapping[str, Ligand]",
    mapping_options: "MappingOptions",
    cache: dict[frozenset[str], int] | None = None,
) -> int:
    """Return the heavy-atom count of the N-way MCS over *names*.

    Parameters
    ----------
    names : Iterable[str]
        Ligand names forming the candidate group.
    ligands : Mapping[str, Ligand]
    mapping_options : MappingOptions
        Supplies the ``FindMCS`` settings, which come from exactly one place; see
        :mod:`rbfenetmap.core.mcs`.
    cache : dict[frozenset[str], int], optional
        Memo keyed by group. Supplied by :func:`core_clusters` so a group evaluated while
        considering one merge is not recomputed while considering the next.

    Returns
    -------
    int
        Heavy atoms in the shared core. A single ligand trivially shares all of itself, so
        it returns its own heavy-atom count rather than zero -- otherwise the very first
        merge of two singletons would be gated against a meaningless baseline.

    Notes
    -----
    Hydrogens are excluded to match every other core and soft-core count in the package
    (see :func:`rbfenetmap.core.descriptors.compute_descriptors`), so a threshold here is
    on the same scale as ``min_core_atoms`` in
    :class:`~rbfenetmap.core.options.SoftcorePolicy`.

    The result never exceeds the heavy-atom count of the smallest member, because a shared
    substructure has to embed in every one of them. That invariant is worth knowing because
    the obvious implementation -- counting atoms of the MCS query molecule -- violates it:
    ``CompareAny`` emits generic query atoms whose atomic number is 0, which read as heavy
    and inflate the count. It is tested in ``test_clustering.py``.
    """
    key = frozenset(names)
    if cache is not None and key in cache:
        return cache[key]

    if not key:
        size = 0
    elif len(key) == 1:
        size = ligands[next(iter(key))].n_heavy
    else:
        ordered = sorted(key)
        pattern = mcs_query_many([ligands[name].mol for name in ordered], mapping_options)
        if pattern is None:
            size = 0
        else:
            # Count heavy atoms of the *match in each real molecule*, not atoms of the query.
            # ``CompareAny`` emits generic query atoms whose ``GetAtomicNum()`` is 0, so
            # counting the pattern would score them as heavy and can report a core larger
            # than the smallest member -- which is impossible, and which silently loosens
            # every threshold keyed on this number.
            #
            # A generic atom may also match a hydrogen in one molecule and a heavy atom in
            # another, so the members can disagree. Taking the minimum is the fail-safe
            # reading: the gate then declines to merge rather than merging on shared
            # structure that is not really there.
            counts = []
            for name in ordered:
                mol = ligands[name].mol
                match = mol.GetSubstructMatch(pattern)
                if not match:
                    counts.append(0)
                    break
                counts.append(sum(1 for index in match if mol.GetAtomWithIdx(index).GetAtomicNum() != 1))
            size = min(counts) if counts else 0

    if cache is not None:
        cache[key] = size
    return size


def core_clusters(
    ligands: "Mapping[str, Ligand]",
    feasible_graph: "nx.Graph",
    policy: "ClusteringPolicy",
    mapping_options: "MappingOptions",
) -> tuple[frozenset[str], ...]:
    """Partition *ligands* into clusters sharing a substantial common core.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
        Every ligand. Any not reachable in *feasible_graph* comes back as a singleton
        cluster rather than being dropped.
    feasible_graph : networkx.Graph
        The pairwise-feasible RBFE graph. Only cluster pairs adjacent here are considered
        for merging, so a cluster is always internally mappable.
    policy : ClusteringPolicy
        The thresholds.
    mapping_options : MappingOptions

    Returns
    -------
    tuple[frozenset[str], ...]
        A partition of every ligand name, ordered largest first then by sorted members, so
        the result is stable across runs and cluster 1 is the dominant scaffold.

    Notes
    -----
    Every ligand lands in exactly one cluster, singletons included. A ligand nothing could
    be mapped to is a cluster of one, which is the honest description: it has no
    sub-network, and it will be reached by a bridge.
    """
    clusters: list[frozenset[str]] = [frozenset({name}) for name in sorted(ligands)]
    cache: dict[frozenset[str], int] = {}

    def admissible(union: frozenset[str]) -> int | None:
        """Return the union's core size if it may merge, else ``None``."""
        if policy.max_cluster_size is not None and len(union) > policy.max_cluster_size:
            return None
        core = cluster_core_size(union, ligands, mapping_options, cache)
        if core < policy.min_core_atoms:
            return None
        smallest = min(ligands[name].n_heavy for name in union) or 1
        if core / smallest < policy.min_core_fraction:
            return None
        return core

    while True:
        best: tuple[int, float, tuple[str, ...]] | None = None
        best_pair: tuple[int, int] | None = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Adjacency in the feasible graph is the precondition for merging: a
                # cluster whose members cannot be mapped to one another has no internal
                # network to build.
                if not any(feasible_graph.has_edge(a, b) for a in clusters[i] for b in clusters[j]):
                    continue
                union = clusters[i] | clusters[j]
                core = admissible(union)
                if core is None:
                    continue
                # Largest shared core wins; ties break on the cheapest connecting edge and
                # then on names, so the partition never depends on iteration order.
                cheapest = min(
                    (
                        feasible_graph.edges[a, b].get("weight", 0.0)
                        for a in clusters[i]
                        for b in clusters[j]
                        if feasible_graph.has_edge(a, b)
                    ),
                    default=0.0,
                )
                key = (-core, cheapest, tuple(sorted(union)))
                if best is None or key < best:
                    best, best_pair = key, (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        merged = clusters[i] | clusters[j]
        logger.debug(
            "Merging %s + %s -> shared core %d heavy atom(s)", sorted(clusters[i]), sorted(clusters[j]), -best[0]
        )
        clusters = [c for index, c in enumerate(clusters) if index not in (i, j)] + [merged]

    ordered = tuple(sorted(clusters, key=lambda c: (-len(c), tuple(sorted(c)))))
    logger.info(
        "Core clustering: %d ligand(s) -> %d cluster(s) %s", len(ligands), len(ordered), [sorted(c) for c in ordered]
    )
    return ordered


def describe_clusters(
    clusters: "Sequence[frozenset[str]]",
    ligands: "Mapping[str, Ligand]",
    mapping_options: "MappingOptions",
    policy: "ClusteringPolicy",
) -> list[str]:
    """Return one human-readable line per cluster, plus any shortfalls.

    Used by the CLI and the report. Separated from :func:`core_clusters` so the partition
    itself stays a pure function of its inputs.
    """
    lines: list[str] = []
    for index, cluster in enumerate(clusters, start=1):
        core = cluster_core_size(cluster, ligands, mapping_options)
        members = sorted(cluster)
        preview = members[:6]
        suffix = "..." if len(members) > 6 else ""
        note = "" if len(cluster) >= policy.min_cluster_size else "  (too small to carry a cycle)"
        lines.append(f"cluster {index} ({len(members)}): core {core} heavy atom(s) {preview}{suffix}{note}")
    return lines
