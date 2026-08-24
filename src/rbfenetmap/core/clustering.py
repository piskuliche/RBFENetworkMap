"""Partitioning a ligand set into clusters, so the network can be planned per cluster.

Why a partition changes the edge budget
---------------------------------------
Pitman *et al.* (*JCIM* 2023, 63, 1776-1793) put the precision floor of an RBFE network at
``k_min ~ n ln n`` edges: below it, precision degrades *worse* as the set grows. That floor
is superlinear, and superlinear costs are exactly the ones a partition beats. For clusters
of sizes ``n_1 ... n_d`` summing to ``n``::

    sum_i n_i ln n_i  <  n ln n

with equality only for a single cluster. Planning each cluster to the floor and joining the
clusters with a handful of bridges therefore buys the same per-cluster precision for
``n ln(n/d)``-ish edges: 100 ligands in five balanced clusters need roughly 190 edges rather
than 460. Even a badly imbalanced split saves 30-50%, because the term that dominates is the
largest cluster and it is still smaller than the whole set.

What a cluster is *not*
-----------------------
Not a feasibility statement. Nothing here consults the soft-core budget, a mapping, or a
rejection: clustering is a **selection-level objective**, in the sense the planning notes
insist on, and it shapes which of the feasible edges are worth spending on rather than which
edges exist. Two ligands in different clusters may well have a perfectly good RBFE mapping
between them; the point is that paying for many such edges buys less than paying for edges
inside a cluster, because the within-cluster edges are the ones a cycle can check.

The three clusterers
--------------------
:func:`cluster_by_charge` is exact and free -- net formal charge is a property of the
molecule, not of a similarity threshold, and a charge-changing edge is the one alchemical
transformation the package already penalises hardest. :func:`cluster_by_scaffold` groups on
the Bemis-Murcko framework, which is what a medicinal chemist means by "series".
:func:`cluster_by_fingerprint` is the general fallback for a set with neither a clean charge
split nor a shared framework.

Every clusterer returns ``{ligand name: cluster index}``, and the indices are canonicalised
by :func:`_label_groups` so that the same ligand set always yields the same numbering
regardless of dictionary order or of how the underlying grouping key sorted.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Hashable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rbfenetmap.core.models import Ligand

__all__ = (
    "CLUSTER_METHODS",
    "DEFAULT_FINGERPRINT_CUTOFF",
    "assign_clusters",
    "cluster_by_charge",
    "cluster_by_fingerprint",
    "cluster_by_scaffold",
    "cluster_edge_budget",
    "cluster_sizes",
)

#: The clustering methods :func:`assign_clusters` understands, ``"none"`` included so the
#: option surface has a single vocabulary. ``"none"`` is a real member rather than a
#: sentinel the caller tests for separately, which keeps the planner's dispatch honest.
CLUSTER_METHODS: tuple[str, ...] = ("none", "charge", "scaffold", "fingerprint")

#: Average-linkage cut, as a Tanimoto *distance*, used when the caller names neither a
#: cluster count nor a cutoff. ``0.6`` is one minus ``0.4``, and ``0.4`` is already this
#: package's stated notion of "similar enough to be worth mapping" -- it is the default
#: ``prefilter_min_tanimoto``. Reusing that number rather than inventing a second one means
#: a user who has tuned the prefilter for their chemistry has tuned this in the same
#: direction, and there is only one similarity threshold in the package to reason about.
DEFAULT_FINGERPRINT_CUTOFF = 0.6


def _label_groups(groups: Mapping[Hashable, Sequence[str]]) -> dict[str, int]:
    """Canonicalise a grouping into ``{ligand name: cluster index}``.

    Clusters are numbered by their alphabetically-first member, not by the grouping key.
    The key differs per clusterer -- an ``int`` charge, a scaffold SMILES, an opaque
    ``fcluster`` label whose value depends on merge order -- and only one of those sorts
    meaningfully. Numbering by membership makes the output of all three comparable and
    stable, which matters because the index ends up in a serialized network and in test
    assertions.

    Parameters
    ----------
    groups : Mapping[Hashable, Sequence[str]]
        Ligand names per grouping key.

    Returns
    -------
    dict[str, int]
        Cluster index per ligand name, contiguous from zero.
    """
    ordered = sorted(groups.values(), key=lambda members: sorted(members))
    return {name: index for index, members in enumerate(ordered) for name in sorted(members)}


def cluster_sizes(partition: Mapping[str, int]) -> list[int]:
    """Return the cluster sizes of *partition*, largest first."""
    counts: dict[int, int] = {}
    for cluster in partition.values():
        counts[cluster] = counts.get(cluster, 0) + 1
    return sorted(counts.values(), reverse=True)


def cluster_edge_budget(partition: Mapping[str, int]) -> dict[str, Any]:
    """Report the precision floor of *partition* against the unclustered floor.

    Both numbers are the Pitman ``n ln n`` floor: one evaluated over the whole set, one
    summed over the clusters. The ratio is the saving clustering buys, and reporting it
    rather than asserting it is deliberate -- a partition into one cluster, or into ``n``
    singletons, saves nothing, and a user who has picked a clusterer that does that should
    be able to see it.

    Parameters
    ----------
    partition : Mapping[str, int]
        Cluster index per ligand name.

    Returns
    -------
    dict[str, Any]
        ``n_ligands``, ``n_clusters``, ``sizes``, ``clustered_floor``
        (``sum_i n_i ln n_i``), ``unclustered_floor`` (``n ln n``), and ``saving``, the
        fraction of the unclustered floor the partition avoids. ``saving`` is ``0.0`` for
        a set too small to have a floor at all.
    """
    sizes = cluster_sizes(partition)
    total = sum(sizes)
    clustered = sum(size * math.log(size) for size in sizes if size > 1)
    unclustered = total * math.log(total) if total > 1 else 0.0
    return {
        "n_ligands": float(total),
        "n_clusters": float(len(sizes)),
        "sizes": list(sizes),
        "clustered_floor": clustered,
        "unclustered_floor": unclustered,
        "saving": 0.0 if unclustered <= 0.0 else 1.0 - clustered / unclustered,
    }


def cluster_by_charge(ligands: Mapping[str, "Ligand"]) -> dict[str, int]:
    """Cluster on net formal charge.

    The one clusterer with no threshold in it. Charge is a property of the molecule rather
    than of a similarity measure, and a net charge change is the transformation this package
    already treats as the most expensive thing that can happen to a still-feasible edge --
    so grouping by charge concentrates the edge budget on the edges whose free energies are
    most trustworthy, and forces the charge-crossing edges to be few and deliberately
    chosen.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]

    Returns
    -------
    dict[str, int]
        Cluster index per ligand name. The clusters are exactly the charge classes.
    """
    groups: dict[Hashable, list[str]] = {}
    for name, ligand in ligands.items():
        groups.setdefault(ligand.charge, []).append(name)
    return _label_groups(groups)


def cluster_by_scaffold(ligands: Mapping[str, "Ligand"]) -> dict[str, int]:
    """Cluster on the Bemis-Murcko scaffold.

    RDKit's ``MurckoScaffold.GetScaffoldForMol`` strips every side chain, leaving the ring
    systems and the linkers between them -- which is close to what a medicinal chemist means
    by "series". Ligands sharing a framework are the ones an MCS search relates cleanly, so
    a scaffold cluster is usually also a well-connected RBFE subnetwork, and the edges the
    partition sacrifices are the scaffold hops that were the least trustworthy anyway.

    An acyclic ligand has an empty scaffold. Those are grouped together under that empty
    scaffold rather than each becoming a singleton: "has no ring system" is a genuine shared
    property here, and scattering them into singletons would produce a partition with as
    many bridges as ligands.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]

    Returns
    -------
    dict[str, int]
        Cluster index per ligand name.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups: dict[Hashable, list[str]] = {}
    for name, ligand in ligands.items():
        scaffold = MurckoScaffold.GetScaffoldForMol(ligand.mol)
        groups.setdefault(Chem.MolToSmiles(scaffold), []).append(name)
    return _label_groups(groups)


def cluster_by_fingerprint(
    ligands: Mapping[str, "Ligand"], *, n_clusters: int | None = None, cutoff: float | None = None
) -> dict[str, int]:
    """Cluster by average-linkage hierarchical clustering on Tanimoto distance.

    Uses :func:`scipy.cluster.hierarchy.linkage` on the condensed ``1 - Tanimoto`` matrix,
    then :func:`~scipy.cluster.hierarchy.fcluster` to cut it. **scipy rather than sklearn**:
    scipy is already a core dependency of this package and sklearn is not, and average
    linkage on Tanimoto distance is the standard chemoinformatics choice -- the density
    methods the neighbouring tools reach for (HDBSCAN in Konnektor, DBSCAN in HiMap) would
    buy a noise label this package has no use for, since every ligand must land in some
    cluster to be planned at all.

    Average linkage specifically, rather than single or complete: single linkage chains a
    congeneric series into one cluster through a string of near-duplicates, and complete
    linkage refuses to admit a ligand that is far from *any* member, which splits a real
    series on its most-substituted compound.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
    n_clusters : int, optional
        Cut the dendrogram to exactly this many clusters. Takes precedence over *cutoff*.
    cutoff : float, optional
        Cut at this Tanimoto *distance*; ligands merged below it share a cluster. Defaults
        to :data:`DEFAULT_FINGERPRINT_CUTOFF` when *n_clusters* is also unset.

    Returns
    -------
    dict[str, int]
        Cluster index per ligand name.

    Raises
    ------
    ValueError
        If *n_clusters* is not positive, or *cutoff* is outside ``[0, 1]``.
    """
    from itertools import combinations

    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage

    from rbfenetmap.core.pairs import fingerprint_pair_similarities

    if n_clusters is not None and n_clusters < 1:
        raise ValueError(f"n_clusters must be at least 1, got {n_clusters}.")
    if cutoff is not None and not 0.0 <= cutoff <= 1.0:
        raise ValueError(f"cutoff is a Tanimoto distance and must lie in [0, 1], got {cutoff}.")

    names = sorted(ligands)
    if len(names) < 2:
        return dict.fromkeys(names, 0)

    # The pair order here is scipy's condensed-matrix order -- (0,1), (0,2), ..., (1,2), ...
    # over the *sorted* names -- so the distances line up with `linkage`'s expectations
    # without any square-matrix round trip.
    pairs = list(combinations(names, 2))
    similarity = fingerprint_pair_similarities(ligands, pairs)
    condensed = np.array([1.0 - similarity[pair] for pair in pairs], dtype=float)

    tree = linkage(condensed, method="average")
    if n_clusters is not None:
        labels = fcluster(tree, t=min(n_clusters, len(names)), criterion="maxclust")
    else:
        labels = fcluster(tree, t=DEFAULT_FINGERPRINT_CUTOFF if cutoff is None else cutoff, criterion="distance")

    groups: dict[Hashable, list[str]] = {}
    for name, label in zip(names, labels):
        groups.setdefault(int(label), []).append(name)
    return _label_groups(groups)


def assign_clusters(ligands: Mapping[str, "Ligand"], method: str, **kwargs: object) -> dict[str, int]:
    """Dispatch to a clusterer by name.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
    method : {"none", "charge", "scaffold", "fingerprint"}
        ``"none"`` puts every ligand in cluster ``0``, which is the partition that makes
        every downstream clustering step a no-op. Returning that rather than raising means
        a caller can hand the option through unconditionally.
    **kwargs
        Forwarded to the chosen clusterer. Only ``"fingerprint"`` accepts any.

    Returns
    -------
    dict[str, int]
        Cluster index per ligand name.

    Raises
    ------
    ValueError
        If *method* is unknown, or a keyword is passed to a clusterer that takes none.
    """
    if method == "none":
        if kwargs:
            raise ValueError(f"cluster_by='none' takes no options; got {sorted(kwargs)}.")
        return dict.fromkeys(sorted(ligands), 0)
    if method == "charge":
        if kwargs:
            raise ValueError(f"cluster_by='charge' takes no options; got {sorted(kwargs)}.")
        return cluster_by_charge(ligands)
    if method == "scaffold":
        if kwargs:
            raise ValueError(f"cluster_by='scaffold' takes no options; got {sorted(kwargs)}.")
        return cluster_by_scaffold(ligands)
    if method == "fingerprint":
        return cluster_by_fingerprint(ligands, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown cluster_by {method!r}. Choose from {list(CLUSTER_METHODS)}.")
