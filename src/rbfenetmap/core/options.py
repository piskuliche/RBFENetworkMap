"""User-tunable options for mapping, repair, scoring, and network selection.

All frozen dataclasses. Conflicting knobs are rejected here, at construction, rather
than deep inside the planner: a user who asks for a connected 12-ligand network with 8
edges should be told immediately, not after the mapping stage has burned several minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rbfenetmap.core.models import EDGE_SEPARATOR, parse_edge_key

__all__ = (
    "ChargeChangePolicy",
    "CorePruningPolicy",
    "EdgeDirection",
    "MappingOptions",
    "NetworkOptions",
    "PairStrategy",
    "RingPolicy",
    "SoftcorePolicy",
    "normalize_edge_specs",
)

RingPolicy = Literal["ring_system", "none"]
ChargeChangePolicy = Literal["allow", "penalize", "reject"]
PairStrategy = Literal["all_unordered_pairs", "all_pairs", "star", "linear", "explicit"]
EdgeDirection = Literal["fewer_softcore_first", "lexicographic", "heavier_second"]


def normalize_edge_specs(specs: tuple[str, ...] | list[str] | None) -> frozenset[tuple[str, str]]:
    """Turn ``"a~b"`` edge specifications into a set of unordered endpoint pairs.

    Selection is undirected -- the free energy of a transformation is antisymmetric, so
    ``a -> b`` and ``b -> a`` name the same experiment. Normalizing forced and banned
    edges to unordered pairs here means a user who writes ``--banned-edge b~a`` gets the
    ban they intended rather than one that silently misses.
    """
    if not specs:
        return frozenset()
    pairs: set[tuple[str, str]] = set()
    for spec in specs:
        source, target = parse_edge_key(spec)
        if source == target:
            raise ValueError(f"Edge specification {spec!r} names the same ligand twice.")
        pairs.add(tuple(sorted((source, target))))  # type: ignore[arg-type]
    return frozenset(pairs)


@dataclass(frozen=True)
class CorePruningPolicy:
    """Which mapped atom pairs to demote before the connectivity repair runs.

    Generalizes ``BuildEdges._classify_softcore_method0/1/2`` from a three-way string
    into independent flags. The named presets reproduce the original three methods.

    Parameters
    ----------
    demote_element_mismatch : bool
        Demote pairs whose atomic numbers differ (the ``MCSS-E2`` behaviour).
    demote_degree_mismatch : bool
        Demote pairs whose heavy-atom connectivity differs (the ``MCSS-E`` behaviour).
    demote_formal_charge_mismatch : bool
        Demote pairs whose formal charges differ. On by default: a charge that changes
        across the core/soft-core boundary is a common source of unphysical setups.
    demote_aromaticity_mismatch, demote_ring_membership_mismatch : bool
        Demote pairs disagreeing on aromaticity or ring membership. Off by default;
        geometry mappers already filter ring/non-ring pairs upstream.
    demote_light_element_swap : bool
        Demote across a hydrogen/heavy-atom pairing, taking the attached branch with it.
    """

    demote_element_mismatch: bool = False
    demote_degree_mismatch: bool = False
    demote_formal_charge_mismatch: bool = True
    demote_aromaticity_mismatch: bool = False
    demote_ring_membership_mismatch: bool = False
    demote_light_element_swap: bool = True

    @classmethod
    def preset(cls, name: str) -> "CorePruningPolicy":
        """Return a named preset: ``"mcss"``, ``"mcss-e"``, or ``"mcss-e2"``."""
        presets = {
            "mcss": cls(demote_element_mismatch=False, demote_degree_mismatch=False),
            "mcss-e": cls(demote_element_mismatch=False, demote_degree_mismatch=True),
            "mcss-e2": cls(demote_element_mismatch=True, demote_degree_mismatch=True),
        }
        if name not in presets:
            raise ValueError(f"Unknown core-pruning preset {name!r}. Choose from {sorted(presets)}.")
        return presets[name]


@dataclass(frozen=True)
class SoftcorePolicy:
    """Controls the soft-core connectivity repair and the feasibility budget.

    Parameters
    ----------
    ring_policy : {"ring_system", "none"}
        ``"ring_system"`` (default) never leaves a ring half soft-core: touching any ring
        atom absorbs the whole ring, and fused systems cascade. ``"none"`` permits
        half-broken rings, which is what a deliberate ring-opening study needs.
    max_softcore_atoms : int
        Reject when either side's heavy soft-core exceeds this. The single most effective
        knob for controlling how aggressive the repair is allowed to be.
    max_softcore_fraction : float
        Reject when either side's heavy soft-core exceeds this fraction of that molecule.
        Catches the small-ligand case that an absolute count misses.
    min_core_atoms : int
        Reject when fewer than this many heavy atoms remain in the common core.
    min_mcs_fraction : float
        Reject before repair when the core covers less than this fraction of the smaller
        molecule. Cheap scaffold-hop filter that avoids wasting Steiner work.
    core_rmsd_threshold : float
        Reject when the mapped core's in-place RMSD exceeds this, in angstroms.
    charge_change_policy : {"allow", "penalize", "reject"}
        How to treat a net formal charge change across the edge.
    max_iterations : int, optional
        Repair loop bound. ``None`` derives it from the molecule sizes, which is already
        a proven upper bound -- this is only a backstop.
    """

    ring_policy: RingPolicy = "ring_system"
    max_softcore_atoms: int = 12
    max_softcore_fraction: float = 0.6
    min_core_atoms: int = 4
    min_mcs_fraction: float = 0.35
    core_rmsd_threshold: float = 2.0
    charge_change_policy: ChargeChangePolicy = "penalize"
    max_iterations: int | None = None
    core_pruning: CorePruningPolicy = field(default_factory=CorePruningPolicy)

    def __post_init__(self) -> None:
        """Reject nonsensical budgets up front."""
        if self.ring_policy not in ("ring_system", "none"):
            raise ValueError(f"Unknown ring_policy {self.ring_policy!r}. Choose 'ring_system' or 'none'.")
        if self.charge_change_policy not in ("allow", "penalize", "reject"):
            raise ValueError(
                f"Unknown charge_change_policy {self.charge_change_policy!r}. Choose 'allow', 'penalize', or 'reject'."
            )
        if self.max_softcore_atoms < 1:
            raise ValueError("max_softcore_atoms must be at least 1.")
        if not 0.0 < self.max_softcore_fraction <= 1.0:
            raise ValueError("max_softcore_fraction must lie in (0, 1].")
        if not 0.0 <= self.min_mcs_fraction <= 1.0:
            raise ValueError("min_mcs_fraction must lie in [0, 1].")
        if self.min_core_atoms < 0:
            raise ValueError("min_core_atoms must be non-negative.")


@dataclass(frozen=True)
class MappingOptions:
    """Controls how a mapper proposes an atom correspondence.

    Parameters
    ----------
    timeout : int
        Seconds allowed for an MCS search.
    ring_matches_ring_only, complete_rings_only : bool
        RDKit ``FindMCS`` settings, mirroring the values ``BuildEdges._find_mcs`` uses.
    match_valences, match_chiral_tag : bool
        Further ``FindMCS`` settings.
    max_matches : int
        Cap on substructure embeddings enumerated when resolving a symmetric core.
    match_selection : {"fewest_fragments", "best_rmsd", "first"}
        How to choose among those embeddings. See the note below.
    distance_threshold : float
        Geometric cutoff, in angstroms, for the geometry-based mappers.
    core_pruning : CorePruningPolicy
        Pre-repair demotions applied to the raw correspondence.

    Notes
    -----
    ``match_selection`` defaults to ``"fewest_fragments"`` rather than to first-match for
    a specific reason. ``BuildEdges._find_mcs`` calls the singular ``GetSubstructMatch``
    on each molecule independently and zips the two results together. For any symmetric
    substructure -- a para-substituted ring being the everyday case -- the two matches can
    correspond to different orientations, and the zip then pairs atoms that sit on
    opposite sides of the ring. The mapping is topologically valid, so nothing complains
    until the geometry check much later. Enumerating embeddings and picking one by an
    explicit criterion removes the coin flip.
    """

    timeout: int = 60
    ring_matches_ring_only: bool = True
    complete_rings_only: bool = True
    match_valences: bool = False
    match_chiral_tag: bool = False
    max_matches: int = 1000
    match_selection: Literal["fewest_fragments", "best_rmsd", "first"] = "fewest_fragments"
    distance_threshold: float = 2.0
    core_pruning: CorePruningPolicy = field(default_factory=CorePruningPolicy)


@dataclass(frozen=True)
class NetworkOptions:
    """Controls candidate generation and final edge selection.

    Parameters
    ----------
    pair_strategy : PairStrategy
        How candidate pairs are enumerated before scoring.
    hub : str, optional
        Ligand to place at the centre of a ``star`` network, or to bias the MST toward.
    explicit_pairs : tuple[str, ...]
        ``"a~b"`` specifications used by the ``explicit`` strategy.
    n_edges : int, optional
        Cap on the total number of selected edges.
    edges_per_ligand : int
        Target minimum degree for every ligand. Best-effort.
    min_cycle_coverage : float
        Target fraction of ligands lying on at least one cycle. Best-effort. Cycles are
        what make a network's free energies checkable against themselves, so this is the
        knob that buys statistical confidence rather than raw coverage.
    forced_edges, banned_edges : tuple[str, ...]
        ``"a~b"`` specifications, normalized to unordered pairs.
    require_connected : bool
        Whether the selected network must span every ligand.
    edge_direction : EdgeDirection
        How each selected edge is oriented once selection is done.
    prefilter : {"none", "fingerprint"}
        Optional similarity prefilter applied before mapping.
    prefilter_k : int
        Neighbours retained per ligand by the prefilter.
    prefilter_min_tanimoto : float
        Similarity floor for the prefilter.
    jobs : int
        Worker processes used for mapping and scoring.
    consistency : {"pairwise", "graph"}
        ``"graph"`` additionally intersects each ligand's core across all its edges.
    softcore : SoftcorePolicy
        Feasibility policy handed to the repair.

    Raises
    ------
    ValueError
        If an edge appears in both the forced and banned sets, or if a knob is
        out of range.
    """

    pair_strategy: PairStrategy = "all_unordered_pairs"
    hub: str | None = None
    explicit_pairs: tuple[str, ...] = ()
    n_edges: int | None = None
    edges_per_ligand: int = 2
    min_cycle_coverage: float = 1.0
    forced_edges: tuple[str, ...] = ()
    banned_edges: tuple[str, ...] = ()
    require_connected: bool = True
    edge_direction: EdgeDirection = "fewer_softcore_first"
    prefilter: Literal["none", "fingerprint"] = "none"
    prefilter_k: int = 8
    prefilter_min_tanimoto: float = 0.4
    jobs: int = 1
    consistency: Literal["pairwise", "graph"] = "pairwise"
    softcore: SoftcorePolicy = field(default_factory=SoftcorePolicy)

    def __post_init__(self) -> None:
        """Reject contradictory or out-of-range settings."""
        forced = self.forced_pairs
        banned = self.banned_pairs
        overlap = sorted(forced & banned)
        if overlap:
            raise ValueError(
                f"Edge(s) {[f'{a}{EDGE_SEPARATOR}{b}' for a, b in overlap]} appear in both forced_edges "
                "and banned_edges. Selection cannot both require and forbid an edge."
            )
        if self.edges_per_ligand < 1:
            raise ValueError("edges_per_ligand must be at least 1.")
        if not 0.0 <= self.min_cycle_coverage <= 1.0:
            raise ValueError("min_cycle_coverage must lie in [0, 1].")
        if self.n_edges is not None and self.n_edges < 1:
            raise ValueError("n_edges must be at least 1 when set.")
        if self.jobs < 1:
            raise ValueError("jobs must be at least 1.")
        if self.pair_strategy == "star" and not self.hub:
            raise ValueError("pair_strategy='star' requires a hub ligand.")
        if self.pair_strategy == "explicit" and not self.explicit_pairs:
            raise ValueError("pair_strategy='explicit' requires explicit_pairs.")

    @property
    def forced_pairs(self) -> frozenset[tuple[str, str]]:
        """Forced edges as unordered endpoint pairs."""
        return normalize_edge_specs(self.forced_edges)

    @property
    def banned_pairs(self) -> frozenset[tuple[str, str]]:
        """Banned edges as unordered endpoint pairs."""
        return normalize_edge_specs(self.banned_edges)

    def check_edge_budget(self, n_ligands: int) -> None:
        """Verify ``n_edges`` can support a spanning network over *n_ligands*.

        Raises
        ------
        ValueError
            If ``n_edges`` is below ``n_ligands - 1`` while connectivity is required.

        Notes
        -----
        This is the single most likely knob conflict, and it is a hard error rather than
        a silent override in either direction. Trimming the spanning tree to honour
        ``n_edges`` would produce a disconnected network the user explicitly forbade;
        quietly raising ``n_edges`` would ignore a budget the user explicitly set. Only
        the user can say which they meant.
        """
        if self.n_edges is None or not self.require_connected or n_ligands < 2:
            return
        minimum = n_ligands - 1
        if self.n_edges < minimum:
            raise ValueError(
                f"n_edges={self.n_edges} cannot connect {n_ligands} ligands; a spanning network needs at "
                f"least {minimum} edges. Raise n_edges to >= {minimum}, or pass require_connected=False."
            )
