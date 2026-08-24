"""User-tunable options for mapping, repair, scoring, and network selection.

All frozen dataclasses. Conflicting knobs are rejected here, at construction, rather
than deep inside the planner: a user who asks for a connected 12-ligand network with 8
edges should be told immediately, not after the mapping stage has burned several minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rbfenetmap.core.clustering import CLUSTER_METHODS
from rbfenetmap.core.intermediates import INTERMEDIATE_MODES, IntermediateOptions
from rbfenetmap.core.models import EDGE_SEPARATOR, parse_edge_key

__all__ = (
    "COMPAT_LEVELS",
    "DESIGN_CRITERIA",
    "CONSISTENCY_SCOPES",
    "AlignmentMethod",
    "DesignCriterion",
    "AlignmentOptions",
    "CBFEMode",
    "ClusterMethod",
    "CompatLevel",
    "ConsistencyScope",
    "ChargeChangePolicy",
    "CorePruningPolicy",
    "CycleCoverageMode",
    "EdgeDirection",
    "HubSelection",
    "PairEvaluation",
    "MappingOptions",
    "NetworkOptions",
    "PairStrategy",
    "RingPolicy",
    "SelectionObjective",
    "SoftcorePolicy",
    "normalize_edge_specs",
)

AlignmentMethod = Literal["mcs", "o3a"]
RingPolicy = Literal["ring_system", "none"]
ChargeChangePolicy = Literal["allow", "penalize", "reject"]
PairStrategy = Literal["all_unordered_pairs", "all_pairs", "star", "linear", "explicit"]
EdgeDirection = Literal["fewer_softcore_first", "lexicographic", "heavier_second"]
SelectionObjective = Literal["uniform_redundancy", "connectivity_then_cycles"]
PairEvaluation = Literal["eager", "adaptive"]
CycleCoverageMode = Literal["node", "edge"]
HubSelection = Literal["most_partners", "min_total_cost"]
CBFEMode = Literal["off", "bridge", "cycles", "all"]
ClusterMethod = Literal["none", "charge", "scaffold", "fingerprint"]
DesignCriterion = Literal["none", "a_optimal", "d_optimal"]
ConsistencyScope = Literal["pairwise", "component", "graph"]

#: How widely one core -- and therefore one soft-core -- is required per ligand, ordered
#: from loosest to strictest. A scope's position is what callers test against, rather than
#: a chain of equality checks that would drift as scopes are added.
CONSISTENCY_SCOPES: tuple[ConsistencyScope, ...] = ("pairwise", "component", "graph")
CompatLevel = Literal["v0.4"]

#: Statistical design criteria, plus the ``"none"`` that means "select on cost alone".
#: ``"none"`` is a member rather than an ``Optional`` because every other selection knob in
#: this class spells "off" as a value, and one knob that spells it as ``None`` would be a
#: second convention for the same idea.
DESIGN_CRITERIA: tuple[DesignCriterion, ...] = ("none", "a_optimal", "d_optimal")

#: Released behaviours a run can be pinned to. Versioned rather than a single ``legacy``
#: flag: "legacy" stops meaning anything the moment there are two of them, and the whole
#: point of the mechanism is to still be unambiguous several releases from now.
COMPAT_LEVELS: tuple[CompatLevel, ...] = ("v0.4",)

#: Ordered from least to most CBFE usage. ``cycles`` includes everything ``bridge`` does,
#: so a mode's position in this tuple is what the planner tests against rather than a
#: chain of equality checks that would drift as modes are added.
CBFE_MODES: tuple[CBFEMode, ...] = ("off", "bridge", "cycles", "all")


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
        Seconds allowed for a single MCS search.

        **This is the memory knob as much as the time knob.** ``FindMCS`` allocates
        monotonically while it searches and frees nothing until it returns, at roughly
        40 MB per second on drug-sized ligands, so peak usage is about
        ``40 MB/s * timeout * jobs``. The default of 60 with ``jobs=8`` is therefore some
        20 GB of search structures before a single candidate is retained. Raise it
        knowingly.
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
class AlignmentOptions:
    """Controls the optional pre-alignment of a ligand set into a common frame.

    Parameters
    ----------
    method : {"mcs", "o3a"}
        ``"mcs"`` (default) fits each ligand onto an already-aligned neighbour through
        their maximum common substructure, which gives an auditable set of atoms and a
        residual RMSD that means something. ``"o3a"`` uses RDKit's Open3DAlign, which
        needs no shared substructure and is the fallback for a set too diverse for an MCS
        to bite on.
    reference : str, optional
        Name of the ligand whose frame everything else is brought into. ``None`` picks the
        ligand with the most heavy atoms, ties broken by name. In a congeneric series the
        largest ligand usually *contains* the shared scaffold, so its substructure overlap
        with every partner is large, and the rule costs no MCS searches to evaluate.
    min_mcs_atoms : int
        Refuse to fit on fewer corresponding atoms than this. A ligand that cannot clear
        the bar is left in its own frame and reported, rather than moved on the strength
        of an overlap too small to determine where it should go.
    max_matches : int
        Cap on the substructure embeddings enumerated while resolving a symmetric overlap.

    Raises
    ------
    ValueError
        If the method is unknown, ``min_mcs_atoms`` is below three, or ``max_matches`` is
        not positive.

    Notes
    -----
    There is deliberately no ``"none"`` method. Alignment is either requested or not
    requested; a do-nothing member would be a second way to express "off" that every
    caller downstream would then have to test for.

    ``max_matches`` defaults well below :class:`MappingOptions`' 1000 because the jobs are
    not comparable. Mapping is choosing the common core an alchemical transformation will
    actually run, once per candidate edge; this is choosing a *frame*, once per ligand, and
    the answer is a rigid motion that a few hundred embeddings pin down as well as a
    thousand would.
    """

    method: AlignmentMethod = "mcs"
    reference: str | None = None
    min_mcs_atoms: int = 3
    max_matches: int = 200

    def __post_init__(self) -> None:
        """Reject nonsensical settings up front."""
        if self.method not in ("mcs", "o3a"):
            raise ValueError(f"Unknown alignment method {self.method!r}. Choose 'mcs' or 'o3a'.")
        if self.min_mcs_atoms < 3:
            raise ValueError(
                f"min_mcs_atoms must be at least 3, got {self.min_mcs_atoms}. Three non-collinear "
                "points is the minimum that fixes a rigid body in space; with fewer, the fit is not "
                "underdetermined so much as meaningless."
            )
        if self.max_matches < 1:
            raise ValueError("max_matches must be at least 1.")


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
    selection_objective : {"uniform_redundancy", "connectivity_then_cycles"}
        Whether redundancy first tries to raise degree targets uniformly, or instead
        focuses on putting as many ligands as possible on at least one cycle after the
        spanning network has been built.
    cycle_coverage_mode : {"node", "edge"}
        What ``min_cycle_coverage`` is a fraction *of*. ``"node"`` (the default, and
        LOMAP's rule) measures the ligands that lie on at least one cycle. ``"edge"``
        measures the selected edges that lie on one, which is FEP+'s stated invariant and
        is exactly 2-edge-connectivity: at coverage 1.0 the network has no bridges at all.
        The edge form is strictly the harder target -- every bridge has covered endpoints
        as soon as something else puts them on a cycle -- so it is opt-in rather than a
        correction to the node form.
    max_cycle_size : int, optional
        Maximum cycle length allowed when adding redundancy edges to improve cycle
        coverage. ``None`` permits any cycle size.
    max_diameter : int, optional
        Target upper bound on the network's diameter -- the longest shortest path between
        any two ligands, counted in edges. Statistical error accumulates along a path, so
        LOMAP caps it at 6 and FEP+ below 5. ``None`` (the default) imposes no bound.

        Best-effort, like the other redundancy targets: selection here is *additive*, so
        the bound is approached by buying shortcut edges rather than, as LOMAP does, by
        refusing to remove one. A pool with no shortcut left to sell warns and records the
        shortfall instead of raising.
    n_redundancy : int
        Number of spanning trees the ``redundant-mst`` planner overlays. Ignored by every
        other planner. Konnektor defaults to 2; the paper that introduced the topology
        uses 3.
    hub_selection : {"most_partners", "min_total_cost"}
        How the ``star`` planner picks a hub when none is named. ``"most_partners"`` (the
        default) ranks by feasible partner count and only breaks ties on cost, so it never
        compares cost across ligands of differing connectivity. ``"min_total_cost"`` ranks
        by summed cost to *every other ligand*, charging an unreachable partner the worst
        cost in the pool; that is LOMAP's ``pick_lead`` and HiMap's ``ref_lig_gen``, which
        sum a similarity matrix in which an unrelatable pair scores zero. OpenEye's own
        documentation calls hub choice the dominant factor in a star map's performance,
        which is why it is a knob rather than a constant.
    pair_evaluation : {"eager", "adaptive"}
        Whether to map every candidate before planning, or evaluate fingerprint-ranked
        batches until the requested network targets are met.
    adaptive_initial_neighbors : int
        Fingerprint-nearest neighbours evaluated per ligand in the first adaptive batch.
    adaptive_batch_size : int
        Maximum number of additional pairs evaluated in each adaptive expansion.
    show_progress : bool
        Write pair-evaluation progress to stderr. Disabled by default for library use;
        the CLI enables it automatically on interactive terminals.
    jobs : int
        Worker processes used for mapping and scoring.
    consistency : {"pairwise", "component", "graph"}
        How widely a ligand is required to hold **one** common core -- and therefore, since
        the core and the soft-core are a strict partition of the ligand's atoms, **one
        soft-core**. That corollary is the point of the knob: the Amber ``scmask`` is the
        soft-core, so a ligand with one core across its edges has one ``scmask`` across
        them too.

        A ladder from loosest to strictest:

        - ``"pairwise"`` (default) -- no requirement. Each edge is mapped independently and
          holds the largest core its own pair supports, which is the cheapest
          transformation for that pair. A ligand on three edges holds three cores, and
          three soft-cores.
        - ``"component"`` -- one core per ligand within each connected component of the
          RBFE-only selected subgraph. Costs nothing to configure and is the scope that
          works on a set whose scaffolds do not all map to each other, because such a set
          fragments into exactly those components anyway.
        - ``"graph"`` -- one core per ligand across *all* of its selected RBFE edges. The
          strongest form, and the one that fails first: the shared core is an intersection
          over the whole network, so one chemically distant ligand shrinks everyone's.

        In every case the surviving core is the *intersection* of the pairwise ones, the
        rest is demoted, and the soft-core repair re-runs on what remains, iterated to a
        fixed point.

        Two limits apply to all of them. **CBFE edges are exempt** -- a counterpoised edge
        has no common core by construction, so reading one as "this ligand's core is empty
        here" would erase the core of every ligand a bridge touches. And the pass never
        re-selects; it can leave a selected edge infeasible, which is raised rather than
        absorbed. See :mod:`rbfenetmap.core.consistency`.
    cbfe_mode : {"off", "bridge", "cycles", "all"}
        How freely the planner may spend counterpoised (CBFE) edges. A CBFE edge needs no
        atom mapping, so it is available between *any* two ligands -- including the pairs
        an MCS search cannot relate -- at the price of two absolute calculations.

        - ``"off"`` -- never. Every edge is RBFE.
        - ``"bridge"`` -- only to join subnetworks the feasible RBFE pool leaves
          disconnected. This is the mode that turns a hard connectivity failure into a
          planned network.
        - ``"cycles"`` -- everything ``"bridge"`` does, and additionally to put ligands on
          a cycle when no RBFE candidate can.
        - ``"all"`` -- the whole network is CBFE. Mapping is skipped entirely.

        The modes form a strict ladder, so raising the setting only ever adds
        possibilities.

        **Eligibility is a gate applied before cost competition, and this is the point
        most easily misread.** ``cbfe_base_cost`` decides *which* CBFE edge is chosen
        among the ones the mode makes eligible, and orders RBFE against CBFE inside cycle
        closure. It never lets a CBFE edge outbid a feasible RBFE edge inside an already
        connected component: under ``"bridge"`` a CBFE edge that does not join two
        components is not in the pool at all, at any price.
    cbfe_base_cost : float
        Fixed cost of a CBFE edge, on the same scale as the scorer's edge totals. The
        default sits at the linear scorer's charge-change ceiling -- a CBFE edge costs
        about what the most expensive thing that can happen to a still-feasible RBFE edge
        costs -- so CBFE never wins on price alone, only on availability.
    cbfe_atom_weight : float
        Added to ``cbfe_base_cost`` for each heavy atom summed over both ligands. A
        counterpoised calculation decouples both molecules in full, so its expense scales
        with how much there is to decouple.
    cluster_by : {"none", "charge", "scaffold", "fingerprint"}
        Partition the ligands and plan each cluster as its own subnetwork, joined to the
        others by a few deliberately chosen edges. ``"none"`` (default) is the unpartitioned
        behaviour.

        This is an **edge-budget** knob, not a feasibility one. The precision floor of an
        RBFE network goes as ``n ln n``, and that is superlinear, so ``sum_i n_i ln n_i`` is
        strictly smaller than ``n ln n`` for any real partition: planning five balanced
        clusters of twenty to the floor costs roughly 190 edges where one hundred ligands
        cost 460. Nothing here changes which edges are *feasible* -- cross-cluster mappings
        are as available as they ever were, they are simply not worth buying in quantity.

        - ``"charge"`` -- net formal charge classes. Exact, thresholdless, and it isolates
          the transformation the scorer already penalises hardest.
        - ``"scaffold"`` -- the Bemis-Murcko framework, which is what "series" usually means.
        - ``"fingerprint"`` -- average-linkage hierarchical clustering on Tanimoto distance,
          for a set with neither a clean charge split nor a shared framework.
    cluster_bridges : int
        Edges spent joining each pair of clusters that gets joined, when ``cluster_by`` is
        set. Ignored otherwise.

        The default of ``2`` is deliberate and is the reason this is not simply ``1``. Two
        edges between the same two clusters put the crossing itself **on a cycle**, since
        the paths inside each cluster close the loop. Cross-cluster edges are the least
        similar and therefore the least trustworthy edges in the network, so applying the
        every-edge-in-a-cycle invariant precisely there buys more per edge than anywhere
        else. Setting ``1`` gives the minimal spanning join and leaves each crossing
        unchecked.
    design : {"none", "a_optimal", "d_optimal"}
        Statistical design criterion the ``optimal`` planner minimises. ``"none"``
        (default) selects on cost alone, which is what every release up to v0.4 did.

        The criterion is evaluated on the network's Fisher information matrix, which for a
        set of relative measurements *is* the weighted graph Laplacian -- see
        :mod:`rbfenetmap.core.design`. ``a_optimal`` minimises the summed variance of the
        estimates; ``d_optimal`` minimises the volume of their joint confidence ellipsoid,
        which because the Laplacian's pseudo-determinant counts spanning trees produces a
        markedly more cyclic network at the same edge count. Prefer ``d_optimal`` when a
        cycle-closure correction will be applied downstream, ``a_optimal`` otherwise.

        This is an *objective*, so it is meaningful only to a planner that optimises it.
        Naming it alongside a planner that does not is refused rather than ignored; see
        :meth:`~rbfenetmap.core.meta.planners.AbstractNetworkPlanner.check_design_support`.
    design_candidate_factor : float
        The design's candidate pool is capped at ``design_candidate_factor * n_ligands``
        edges -- the ``M = 3m`` of Xu's Appendix-H heuristic. Raising it widens the search
        at quadratic cost in criterion evaluations; the published 1.10x bound is measured at
        the default of 3.0.
    design_refine : bool
        Run a Fedorov exchange pass after the heuristic, swapping edges one at a time while
        the criterion improves. Off by default: the heuristic is already within a published
        1.10x of the optimum and the refinement costs far more criterion evaluations.
    design_total_ns : float, optional
        Total simulation budget, in nanoseconds, to distribute A-optimally across the
        selected edges. ``None`` (default) emits no allocation at all. Set it and the Amber
        exporter writes a per-edge lambda-window and nanosecond budget into each
        ``.runconfig``. **Static, computed once from the predicted variances** -- the
        iterative refit against measured variances needs a round trip through the MD engine
        and is not part of this.
    design_lambda_min, design_lambda_max : int
        Bounds on the per-edge lambda-window count the allocation is mapped onto. The
        defaults, 12 and 24, bracket what an Amber RBFE edge normally runs at.
    softcore : SoftcorePolicy
        Feasibility policy handed to the repair.
    intermediates : IntermediateOptions
        Whether the pipeline may *invent* ligands to bridge pairs no mapping can relate,
        and how many. Off by default.

        This is the only knob in the class that changes the **vertex** set rather than the
        edge set, which is why it sits between ``max_softcore_atoms`` and ``cbfe_mode`` in
        the precedence table: it widens the pool the planner is handed, and it does so
        before the planner runs, so a gap an intermediate closed is simply not a gap by
        the time CBFE eligibility is evaluated. That ordering is the whole of "stay
        relative, then fall back to counterpoised" -- there is no precedence flag behind
        it.
    compat : str, optional
        The released behaviour this run was pinned to, or ``None``. Set by
        :meth:`preset`; recorded so a planned network states which behaviour produced
        it. Purely a label -- it changes nothing on its own, because :meth:`preset` has
        already written the values it stands for.

    Raises
    ------
    ValueError
        If an edge appears in both the forced and banned sets, if a knob is
        out of range, or if ``compat`` names an unknown level.
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
    selection_objective: SelectionObjective = "uniform_redundancy"
    cycle_coverage_mode: CycleCoverageMode = "node"
    max_cycle_size: int | None = None
    max_diameter: int | None = None
    n_redundancy: int = 2
    hub_selection: HubSelection = "most_partners"
    pair_evaluation: PairEvaluation = "eager"
    adaptive_initial_neighbors: int = 3
    adaptive_batch_size: int = 32
    show_progress: bool = False
    jobs: int = 1
    consistency: ConsistencyScope = "pairwise"
    cbfe_mode: CBFEMode = "off"
    cbfe_base_cost: float = 8.0
    cbfe_atom_weight: float = 0.05
    cluster_by: ClusterMethod = "none"
    cluster_bridges: int = 2
    design: DesignCriterion = "none"
    design_candidate_factor: float = 3.0
    design_refine: bool = False
    design_total_ns: float | None = None
    design_lambda_min: int = 12
    design_lambda_max: int = 24
    softcore: SoftcorePolicy = field(default_factory=SoftcorePolicy)
    intermediates: IntermediateOptions = field(default_factory=IntermediateOptions)
    compat: str | None = None

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
        if self.selection_objective not in ("uniform_redundancy", "connectivity_then_cycles"):
            raise ValueError("selection_objective must be 'uniform_redundancy' or 'connectivity_then_cycles'.")
        if self.cycle_coverage_mode not in ("node", "edge"):
            raise ValueError(f"cycle_coverage_mode must be 'node' or 'edge'; got {self.cycle_coverage_mode!r}.")
        if self.max_cycle_size is not None and self.max_cycle_size < 3:
            raise ValueError("max_cycle_size must be at least 3 when set.")
        if self.max_diameter is not None and self.max_diameter < 1:
            raise ValueError("max_diameter must be at least 1 when set; a two-ligand network already has diameter 1.")
        if self.n_redundancy < 1:
            raise ValueError("n_redundancy must be at least 1; one overlaid tree is a plain spanning tree.")
        if self.hub_selection not in ("most_partners", "min_total_cost"):
            raise ValueError(f"hub_selection must be 'most_partners' or 'min_total_cost'; got {self.hub_selection!r}.")
        if self.consistency not in CONSISTENCY_SCOPES:
            raise ValueError(f"consistency must be one of {list(CONSISTENCY_SCOPES)}; got {self.consistency!r}.")
        if self.pair_evaluation not in ("eager", "adaptive"):
            raise ValueError("pair_evaluation must be 'eager' or 'adaptive'.")
        if self.adaptive_initial_neighbors < 1:
            raise ValueError("adaptive_initial_neighbors must be at least 1.")
        if self.adaptive_batch_size < 1:
            raise ValueError("adaptive_batch_size must be at least 1.")
        if self.jobs < 1:
            raise ValueError("jobs must be at least 1.")
        if self.cbfe_mode not in CBFE_MODES:
            raise ValueError(f"cbfe_mode must be one of {list(CBFE_MODES)}; got {self.cbfe_mode!r}.")
        if self.cbfe_base_cost < 0:
            raise ValueError("cbfe_base_cost must not be negative.")
        if self.cbfe_atom_weight < 0:
            raise ValueError("cbfe_atom_weight must not be negative.")
        if self.cluster_by not in CLUSTER_METHODS:
            raise ValueError(f"cluster_by must be one of {list(CLUSTER_METHODS)}; got {self.cluster_by!r}.")
        if self.cluster_bridges < 1:
            raise ValueError(
                "cluster_bridges must be at least 1: joining two clusters takes an edge. Use "
                "cluster_by='none' to plan the ligands as a single set."
            )
        if self.design not in DESIGN_CRITERIA:
            raise ValueError(f"design must be one of {list(DESIGN_CRITERIA)}; got {self.design!r}.")
        if self.design_candidate_factor < 1.0:
            raise ValueError(
                "design_candidate_factor must be at least 1.0; a pool smaller than the ligand count "
                "cannot even hold a spanning tree."
            )
        if self.design_total_ns is not None and not self.design_total_ns > 0:
            raise ValueError("design_total_ns must be positive when set.")
        if self.design_lambda_min < 2:
            raise ValueError("design_lambda_min must be at least 2; a lambda schedule needs two end states.")
        if self.design_lambda_max < self.design_lambda_min:
            raise ValueError(
                f"design_lambda_max={self.design_lambda_max} is below design_lambda_min={self.design_lambda_min}."
            )
        if self.intermediates.mode not in INTERMEDIATE_MODES:
            raise ValueError(
                f"intermediates.mode must be one of {list(INTERMEDIATE_MODES)}; got {self.intermediates.mode!r}."
            )
        if self.compat is not None and self.compat not in COMPAT_LEVELS:
            raise ValueError(f"Unknown compat level {self.compat!r}. Known: {list(COMPAT_LEVELS)}.")
        if self.pair_strategy == "star" and not self.hub:
            raise ValueError("pair_strategy='star' requires a hub ligand.")
        if self.pair_strategy == "explicit" and not self.explicit_pairs:
            raise ValueError("pair_strategy='explicit' requires explicit_pairs.")

    @classmethod
    def preset(cls, level: str, **overrides: object) -> "NetworkOptions":
        """Return the options a released version of the package planned with.

        Parameters
        ----------
        level : str
            A member of :data:`COMPAT_LEVELS`.
        **overrides
            Applied on top of the pinned values. Intended for the settings that describe
            *this run* rather than *this behaviour* -- the ligand-specific intent
            (``hub``, ``forced_edges``, ``banned_edges``, ``explicit_pairs``) and the
            operational knobs (``jobs``, ``show_progress``). Overriding an algorithmic
            knob is permitted here and rejected at the CLI, where the user's intent is
            unambiguous enough to call it a contradiction.

        Returns
        -------
        NetworkOptions
            With :attr:`compat` set to *level*.

        Raises
        ------
        ValueError
            If *level* is unknown.

        Notes
        -----
        **Every value below is written out literally, and that is the entire point.**
        Building this from the dataclass defaults would be shorter and would defeat the
        mechanism: the moment a later release moves a default, the preset would move with
        it and silently stop reproducing the version it names. These numbers are a record
        of what v0.4.0 did, not a view onto what the current code does, so they must be
        edited only to fix a transcription error -- never to track a new default.

        The pinned surface is the *algorithmic* one. Ligand-specific intent is not pinned:
        banning an edge or naming a hub is a statement about one ligand set, not about a
        version's behaviour, so those stay available alongside a compat level.
        """
        if level not in COMPAT_LEVELS:
            raise ValueError(f"Unknown compat level {level!r}. Known: {list(COMPAT_LEVELS)}.")

        pinned: dict[str, object] = {
            "pair_strategy": "all_unordered_pairs",
            "n_edges": None,
            "edges_per_ligand": 2,
            "min_cycle_coverage": 1.0,
            "require_connected": True,
            "edge_direction": "fewer_softcore_first",
            "prefilter": "none",
            "prefilter_k": 8,
            "prefilter_min_tanimoto": 0.4,
            "selection_objective": "uniform_redundancy",
            # v0.4.0 measured cycle coverage over ligands and had no diameter bound, no
            # overlaid-MST planner, and one hub rule. The knobs did not exist, so the value
            # that reproduces the version is whichever one makes each a no-op -- not
            # whichever one a later release settles on as its default.
            "cycle_coverage_mode": "node",
            "max_cycle_size": None,
            "max_diameter": None,
            "n_redundancy": 2,
            "hub_selection": "most_partners",
            "pair_evaluation": "eager",
            "adaptive_initial_neighbors": 3,
            "adaptive_batch_size": 32,
            "consistency": "pairwise",
            "cbfe_mode": "off",
            "cbfe_base_cost": 8.0,
            "cbfe_atom_weight": 0.05,
            "cluster_by": "none",
            "cluster_bridges": 2,
            # v0.4.0 had no statistical design at all, so the pin is the no-op. It must
            # stay "none" even if a later release makes a criterion the default -- that is
            # the whole contract, and this is the knob most likely to test it, since the
            # design planner is also where an n_edges default of round(n ln n) would land.
            "design": "none",
            "design_candidate_factor": 3.0,
            "design_refine": False,
            "design_total_ns": None,
            "design_lambda_min": 12,
            "design_lambda_max": 24,
            "softcore": SoftcorePolicy(
                ring_policy="ring_system",
                max_softcore_atoms=12,
                max_softcore_fraction=0.6,
                min_core_atoms=4,
                min_mcs_fraction=0.35,
                core_rmsd_threshold=2.0,
                charge_change_policy="penalize",
                max_iterations=None,
                core_pruning=CorePruningPolicy(
                    demote_element_mismatch=False,
                    demote_degree_mismatch=False,
                    demote_formal_charge_mismatch=True,
                    demote_aromaticity_mismatch=False,
                    demote_ring_membership_mismatch=False,
                    demote_light_element_swap=True,
                ),
            ),
            # v0.4 could not invent a ligand at all, so the pinned value is generation
            # switched off. Pinning it rather than leaving it to the field default is what
            # keeps ``--compat v0.4`` reproducing v0.4 on the day the default moves.
            "intermediates": IntermediateOptions(
                mode="off",
                generator="pairmap",
                max_intermediates=None,
                max_gaps=None,
                max_molecules=4,
                seed=0xF00D,
                max_pose_attempts=10,
                pose_rmsd_factor=0.5,
                min_link_score=0.2,
                max_dist=3,
                max_cycle=4,
                max_subgraph_dist=4,
                beta=0.1,
            ),
        }
        pinned.update(overrides)
        pinned["compat"] = level
        return cls(**pinned)  # type: ignore[arg-type]

    @property
    def forced_pairs(self) -> frozenset[tuple[str, str]]:
        """Forced edges as unordered endpoint pairs."""
        return normalize_edge_specs(self.forced_edges)

    @property
    def banned_pairs(self) -> frozenset[tuple[str, str]]:
        """Banned edges as unordered endpoint pairs."""
        return normalize_edge_specs(self.banned_edges)

    @property
    def cbfe_bridges_components(self) -> bool:
        """Whether CBFE edges may join otherwise-disconnected subnetworks."""
        return self.cbfe_mode in ("bridge", "cycles")

    @property
    def cbfe_closes_cycles(self) -> bool:
        """Whether CBFE edges may be spent putting ligands on a cycle."""
        return self.cbfe_mode == "cycles"

    @property
    def generates_intermediates(self) -> bool:
        """Whether the pipeline may invent ligands for this run."""
        return self.intermediates.enabled

    def intermediate_headroom(self, n_ligands: int) -> int | None:
        """Return how many ligands may be invented before ``n_edges`` runs out.

        Parameters
        ----------
        n_ligands : int
            Size of the *real* ligand set.

        Returns
        -------
        int or None
            ``None`` when ``n_edges`` is unset, so nothing constrains generation here.
            Otherwise ``n_edges - (n_ligands - 1)``, possibly zero or negative.

        Notes
        -----
        **The budget is spent, never inflated.** Every invented ligand is another vertex,
        so a spanning network over the augmented set needs one more edge than it did
        before. Quietly raising ``n_edges`` to pay for a molecule the user never asked for
        would be exactly the silent over-spend :meth:`check_edge_budget` refuses to make
        in the other direction. When the headroom runs out, generation stops and says so
        on ``unmet_constraints``; it does not raise, because unlike a spanning tree that
        cannot fit, an intermediate that cannot fit leaves a perfectly valid network.
        """
        if self.n_edges is None:
            return None
        return self.n_edges - max(n_ligands - 1, 0)

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

        With ``cbfe_mode`` bridging enabled this becomes the *only* way a spanning network
        can fail. A CBFE edge exists between every pair, so the candidate pool can no
        longer be too sparse to connect the ligands -- only an edge budget below
        ``n_ligands - 1``, or a ban on every bridging pair, can prevent it.
        """
        if self.n_edges is None or not self.require_connected or n_ligands < 2:
            return
        minimum = n_ligands - 1
        if self.n_edges < minimum:
            raise ValueError(
                f"n_edges={self.n_edges} cannot connect {n_ligands} ligands; a spanning network needs at "
                f"least {minimum} edges. Raise n_edges to >= {minimum}, or pass require_connected=False."
            )
