"""The pipeline: map, repair, score, bridge, plan.

:func:`build_network` is the package's main entry point. Everything the CLI does, and
everything an embedding program needs, goes through here.

The stage that most shapes the result is the second one. A mapper is allowed to return a
fragmented soft-core; the repair either fixes it or rejects the edge. Rejection is a
normal outcome recorded on the candidate, never an exception -- one impossible pair among
several hundred must not abort a run.

The fourth stage, :func:`augment_with_intermediates`, is the newest and the only one that
changes the *vertex* set. It is off by default and runs between scoring and planning,
where it can see which pairs the first three stages could not relate and hand exactly
those to a generator. Why there rather than per-pair, inside :func:`build_candidate`, or
as a second pass over an augmented ligand set is argued in that function's docstring; the
short version is that whether an intermediate is worth making is a question about the
*network*, and only a stage that runs once over the settled pool can answer it.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import networkx as nx

from rbfenetmap.core.cbfe import build_cbfe_pool, make_cbfe_transformation
from rbfenetmap.core.consistency import maybe_apply_graph_consistency
from rbfenetmap.core.descriptors import compute_descriptors
from rbfenetmap.core.exceptions import MappingError, NetworkPlanError, RepairError
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.meta.planners import AbstractNetworkPlanner
from rbfenetmap.core.meta.scorers import AbstractScorer
from rbfenetmap.core.models import (
    AtomMapping,
    EdgeScore,
    IntermediateRecord,
    Ligand,
    Network,
    RejectionReason,
    SoftcoreRepair,
    Transformation,
)
from rbfenetmap.core.options import MappingOptions, NetworkOptions
from rbfenetmap.core.intermediates import reserve_intermediate_names
from rbfenetmap.core.pairs import fingerprint_pair_similarities, generate_candidate_pairs
from rbfenetmap.core.softcore import precheck_mapping, repair_softcore_connectivity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rbfenetmap.core.intermediates import IntermediateProposal
    from rbfenetmap.core.meta.intermediates import AbstractIntermediateGenerator

__all__ = (
    "AugmentationResult",
    "augment_with_intermediates",
    "build_candidate",
    "build_network",
    "evaluate_pairs",
    "evaluate_pairs_adaptively",
    "feasible_graph",
)

logger = logging.getLogger(__name__)


class _PairProgress:
    """Small dependency-free stderr progress display for expensive pair mappings."""

    def __init__(self, total: int, *, enabled: bool, label: str = "Mapping pairs") -> None:
        self.total = total
        self.enabled = enabled
        self.label = label
        self.completed = 0
        self.started = time.monotonic()
        self.last_rendered = 0.0

    def __enter__(self) -> "_PairProgress":
        if self.enabled:
            self._render(force=True)
        return self

    def update(self, count: int = 1) -> None:
        """Advance by *count*, throttling terminal redraws."""
        self.completed += count
        self._render(force=self.completed >= self.total)

    def set_label(self, label: str) -> None:
        """Change the phase label and redraw immediately."""
        self.label = label
        self._render(force=True)

    def _render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_rendered < 0.1:
            return
        elapsed = max(now - self.started, 1e-9)
        fraction = self.completed / self.total if self.total else 1.0
        width = 24
        filled = min(width, int(width * fraction))
        bar = "#" * filled + "-" * (width - filled)
        rate = self.completed / elapsed
        remaining = max(self.total - self.completed, 0)
        eta = remaining / rate if rate else 0.0
        sys.stderr.write(
            f"\r{self.label} [{bar}] {self.completed}/{self.total} "
            f"({fraction:6.1%}) {elapsed:6.1f}s elapsed {rate:5.2f}/s ETA {eta:6.1f}s"
        )
        sys.stderr.flush()
        self.last_rendered = now

    def close(self) -> None:
        """Finish the current terminal line, including an early-stop marker."""
        if not self.enabled:
            return
        self._render(force=True)
        suffix = "" if self.completed >= self.total else " (stopped early)"
        sys.stderr.write(f"{suffix}\n")
        sys.stderr.flush()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


def build_candidate(
    source: Ligand,
    target: Ligand,
    mapper: AbstractMapper,
    scorer: AbstractScorer,
    mapping_options: MappingOptions,
    network_options: NetworkOptions,
) -> Transformation:
    """Map, repair, and score a single candidate pair.

    Parameters
    ----------
    source, target : Ligand
    mapper : AbstractMapper
    scorer : AbstractScorer
    mapping_options : MappingOptions
    network_options : NetworkOptions
        Contributes the :class:`~rbfenetmap.core.options.SoftcorePolicy`.

    Returns
    -------
    Transformation
        Always a transformation, never an exception. A pair that cannot be mapped or
        repaired comes back marked infeasible with the reason attached, so it stays
        visible in the audit trail and can explain a later disconnection.
    """
    policy = network_options.softcore
    empty = AtomMapping.from_core_pairs({}, n_atoms_1=source.n_atoms, n_atoms_2=target.n_atoms, method=mapper.name)

    def rejected(reason: RejectionReason, mapping: AtomMapping, repair: SoftcoreRepair) -> Transformation:
        """Assemble an infeasible candidate."""
        descriptors = compute_descriptors(source, target, mapping, repair)
        return Transformation(
            source=source.name,
            target=target.name,
            mapping=mapping,
            repair=repair,
            score=scorer.score_edge(descriptors, rejections=[reason]),
        )

    if not mapper.supports_pair(source, target):
        return rejected(
            RejectionReason.MAPPER_FAILED,
            empty,
            SoftcoreRepair(rejection=RejectionReason.MAPPER_FAILED, trace=("mapper declined the pair",)),
        )

    try:
        mapping = mapper.map_pair(source, target, mapping_options)
    except MappingError as exc:
        logger.debug("%s~%s: mapping failed: %s", source.name, target.name, exc)
        return rejected(
            RejectionReason.MAPPER_FAILED,
            empty,
            SoftcoreRepair(rejection=RejectionReason.MAPPER_FAILED, trace=(str(exc),)),
        )

    reason = precheck_mapping(source, target, mapping, policy)
    if reason is not None:
        return rejected(reason, mapping, SoftcoreRepair(rejection=reason, trace=(f"precheck: {reason.value}",)))

    try:
        repaired, repair = repair_softcore_connectivity(source, target, mapping, policy)
    except RepairError as exc:
        logger.debug("%s~%s: repair failed: %s", source.name, target.name, exc)
        return rejected(
            RejectionReason.SOFTCORE_FRAGMENTED,
            mapping,
            SoftcoreRepair(rejection=RejectionReason.SOFTCORE_FRAGMENTED, trace=(str(exc),)),
        )

    descriptors = compute_descriptors(source, target, repaired, repair)

    # The geometry check runs on the repaired core, so it judges what would actually be
    # held fixed rather than whatever the mapper first proposed.
    rejections: list[RejectionReason] = []
    if repair.rejection is not None:
        rejections.append(repair.rejection)
    elif descriptors["core_rmsd"] > policy.core_rmsd_threshold:
        rejections.append(RejectionReason.CORE_GEOMETRY_MISMATCH)

    score: EdgeScore = scorer.score_edge(descriptors, rejections=rejections)
    return Transformation(source=source.name, target=target.name, mapping=repaired, repair=repair, score=score)


def _evaluate_one(args: tuple) -> Transformation:
    """Evaluate one prepared pair payload."""
    return build_candidate(*args)


def evaluate_pairs(
    ligands: Mapping[str, Ligand],
    pairs: Sequence[tuple[str, str]],
    mapper: AbstractMapper,
    scorer: AbstractScorer,
    mapping_options: MappingOptions,
    network_options: NetworkOptions,
    *,
    progress_callback: Callable[[int], None] | None = None,
) -> list[Transformation]:
    """Map, repair, and score every pair.

    Parallelised over ``network_options.jobs`` *threads*, which keeps the immutable
    ligand and scorer mappings shared by reference; Python process pools cannot
    serialize them.

    Threads win only on the native part of the work -- ``FindMCS`` and the substructure
    search. Core selection, pruning, soft-core repair, and descriptors are pure Python
    and hold the GIL, so scaling is sublinear and flattens well below the core count.
    """
    work = [
        (ligands[source], ligands[target], mapper, scorer, mapping_options, network_options) for source, target in pairs
    ]
    results: list[Transformation | None] = [None] * len(work)
    own_progress = _PairProgress(len(work), enabled=network_options.show_progress and progress_callback is None)
    notify = progress_callback or own_progress.update
    with own_progress:
        if network_options.jobs > 1 and len(work) > 1:
            with ThreadPoolExecutor(max_workers=network_options.jobs) as pool:
                futures = {pool.submit(_evaluate_one, item): index for index, item in enumerate(work)}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    notify(1)
        else:
            for index, item in enumerate(work):
                results[index] = build_candidate(*item)
                notify(1)
    return [result for result in results if result is not None]


def feasible_graph(names: Sequence[str], candidates: Sequence[Transformation]) -> nx.Graph:
    """Build the undirected graph of candidates that passed feasibility checks.

    Parameters
    ----------
    names : Sequence[str]
        Every ligand, so an isolated one is a node of its own rather than absent.
    candidates : Sequence[Transformation]
        Scored candidates, feasible or not.

    Returns
    -------
    networkx.Graph

    Notes
    -----
    Public because two stages now need exactly this graph and they must agree on it: the
    adaptive loop decides which pairs still cross a component boundary, and
    :func:`augment_with_intermediates` decides which gaps are worth offering to a
    generator. Two private copies that drifted apart would show up as a generator being
    offered a gap that no longer exists.
    """
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(names)
    graph.add_edges_from(candidate.unordered_key for candidate in candidates if candidate.feasible)
    return graph


def _initial_adaptive_pairs(
    names: Sequence[str],
    ranked_pairs: Sequence[tuple[str, str]],
    forced_pairs: frozenset[tuple[str, str]],
    neighbors: int,
) -> list[tuple[str, str]]:
    """Seed adaptive evaluation with forced edges and each ligand's nearest neighbours."""
    chosen: set[tuple[str, str]] = set(forced_pairs)
    for name in names:
        incident = [pair for pair in ranked_pairs if name in pair]
        chosen.update(incident[:neighbors])
    return [pair for pair in ranked_pairs if pair in chosen]


def _adaptive_candidate_pool(
    ligands: Mapping[str, Ligand],
    pairs: Sequence[tuple[str, str]],
    mapper: AbstractMapper,
    scorer: AbstractScorer,
    planner: AbstractNetworkPlanner,
    mapping_options: MappingOptions,
    network_options: NetworkOptions,
) -> list[Transformation]:
    """Evaluate promising pairs in batches until the network targets are met.

    Returns the candidate pool rather than a planned network, because two callers need
    different things from it: :func:`evaluate_pairs_adaptively` plans immediately, while
    :func:`build_network` has an intermediate-generation stage to run over the settled
    pool first. Splitting it here rather than re-planning afterwards is what keeps
    generation from seeing a pool the loop was still growing.

    Connectivity expansion always prioritizes unevaluated pairs crossing the current
    feasible components. If connectivity remains impossible, every possible bridge is
    eventually evaluated before the planner reports failure. Once connected, additional
    batches favour deficient ligands and edges that form short cycles.

    Notes
    -----
    Both interactions with ``cbfe_mode`` are deliberate, and they go opposite ways.

    The connectivity branch is left alone. It keys on the *feasible RBFE graph*, so with
    ``cbfe_mode="bridge"`` it still exhausts every cross-component pair before giving up --
    which is exactly what that mode means. A user might reasonably expect enabling CBFE to
    make the search stop sooner; it does not, because the only way to know RBFE cannot
    reach across a gap is to try.

    The intermediate plan, by contrast, is probed with CBFE switched *off*. Left on, a
    ``cycles``-mode plan would satisfy ``min_cycle_coverage`` with counterpoised edges,
    come back with no unmet constraints, and stop the loop -- so CBFE would mask the
    shortfall and short-circuit the very RBFE expansion this function exists to drive. The
    probe also drops ``require_connected``, because it can be reached with a pool no
    further mapping can connect, where the raise is the intended outcome only for the final
    plan. That final plan, below, uses the caller's options unchanged.
    """
    names = list(ligands)
    probe_options = (
        network_options
        if network_options.cbfe_mode == "off"
        else replace(network_options, cbfe_mode="off", require_connected=False)
    )
    similarities = fingerprint_pair_similarities(ligands, pairs)
    ranked_pairs = sorted(pairs, key=lambda pair: (-similarities[pair], pair))
    initial = _initial_adaptive_pairs(
        names, ranked_pairs, network_options.forced_pairs, network_options.adaptive_initial_neighbors
    )
    remaining = [pair for pair in ranked_pairs if pair not in set(initial)]
    candidates: list[Transformation] = []

    with _PairProgress(len(pairs), enabled=network_options.show_progress) as progress:
        batch_number = 0

        def evaluate(batch: Sequence[tuple[str, str]]) -> None:
            nonlocal batch_number
            if not batch:
                return
            batch_number += 1
            progress.set_label(f"Mapping pairs (batch {batch_number}: {len(batch)})")
            logger.info(
                "Adaptive evaluation: mapping %d pair(s), %d previously evaluated, %d remaining",
                len(batch),
                len(candidates),
                len(remaining),
            )
            candidates.extend(
                evaluate_pairs(
                    ligands, batch, mapper, scorer, mapping_options, network_options, progress_callback=progress.update
                )
            )

        evaluate(initial)
        last_network: Network | None = None
        while True:
            components_graph = feasible_graph(names, candidates)
            components = list(nx.connected_components(components_graph))
            membership = {name: index for index, component in enumerate(components) for name in component}
            bridges = [pair for pair in remaining if membership[pair[0]] != membership[pair[1]]]

            # Connectivity is the first objective even when disconnected output is allowed.
            # Trying all current cross-component pairs is also what makes a later failure
            # conclusive rather than an artefact of fingerprint ranking.
            if len(components) > 1 and bridges:
                batch = bridges[: network_options.adaptive_batch_size]
            else:
                # Intermediate planner warnings are expected while the candidate pool is
                # still growing; only warnings from the final returned plan are useful.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    last_network = planner.plan(ligands, candidates, probe_options)

                if not last_network.unmet_constraints or not remaining:
                    break
                if network_options.n_edges is not None and len(last_network.edges) >= network_options.n_edges:
                    break

                selected = last_network.to_networkx()
                degrees = dict(selected.degree())

                def expansion_rank(pair: tuple[str, str]) -> tuple[int, int, float, tuple[str, str]]:
                    deficient = sum(degrees.get(node, 0) < network_options.edges_per_ligand for node in pair)
                    try:
                        cycle_size = nx.shortest_path_length(selected, pair[0], pair[1]) + 1
                    except nx.NetworkXNoPath:
                        cycle_size = len(names) + 1
                    if network_options.max_cycle_size is not None and cycle_size > network_options.max_cycle_size:
                        cycle_size = len(names) + cycle_size
                    return (-deficient, cycle_size, -similarities[pair], pair)

                batch = sorted(remaining, key=expansion_rank)[: network_options.adaptive_batch_size]

            evaluate(batch)
            chosen = set(batch)
            remaining = [pair for pair in remaining if pair not in chosen]

    logger.info("Adaptive evaluation stopped after %d of %d candidate pair(s)", len(candidates), len(pairs))
    return candidates


def evaluate_pairs_adaptively(
    ligands: Mapping[str, Ligand],
    pairs: Sequence[tuple[str, str]],
    mapper: AbstractMapper,
    scorer: AbstractScorer,
    planner: AbstractNetworkPlanner,
    mapping_options: MappingOptions,
    network_options: NetworkOptions,
) -> Network:
    """Evaluate pairs adaptively and plan over what was evaluated.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
    pairs : Sequence[tuple[str, str]]
        Every pair the loop is allowed to reach for, in any order.
    mapper : AbstractMapper
    scorer : AbstractScorer
    planner : AbstractNetworkPlanner
        Drives the loop as well as producing the result: the probe plans are what tell the
        loop whether any target is still unmet.
    mapping_options : MappingOptions
    network_options : NetworkOptions

    Returns
    -------
    Network

    Notes
    -----
    A thin wrapper over :func:`_adaptive_candidate_pool`, which does the work. It stays
    public and keeps returning a :class:`~rbfenetmap.core.models.Network` because that is
    its released signature; handing back a candidate list instead would be a real API
    break for anything that calls it directly.

    Intermediate generation is deliberately **not** run here. It belongs after the loop
    settles and before planning, which is :func:`build_network`'s job -- and a caller who
    reaches for this function directly is asking for adaptive evaluation of the pairs it
    was given, not for new vertices it never mentioned.

    The final plan runs outside the loop's warning suppression, so any genuinely unmet
    best-effort target is visible exactly once. For a required but impossible connection
    it raises with diagnostics, after every component-bridging possibility was attempted.
    """
    candidates = _adaptive_candidate_pool(ligands, pairs, mapper, scorer, planner, mapping_options, network_options)
    return planner.plan(ligands, candidates, network_options)


def _all_cbfe_candidates(ligands: Mapping[str, Ligand], network_options: NetworkOptions) -> tuple[Transformation, ...]:
    """Build the counterpoised candidate pool for ``cbfe_mode="all"``.

    Unlike the bridging pool, these *are* materialized as transformations: they are the
    only candidates the planner will ever see, so they have to occupy the place the mapped
    ones normally would -- including on :attr:`~rbfenetmap.core.models.Network.candidates`,
    where they are the honest audit trail for a network in which nothing was mapped.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
    network_options : NetworkOptions

    Returns
    -------
    tuple[Transformation, ...]
    """
    pool = build_cbfe_pool(ligands, network_options)
    return tuple(make_cbfe_transformation(ligands[a], ligands[b], network_options) for a, b in sorted(pool))


@dataclass(frozen=True)
class AugmentationResult:
    """What intermediate generation added to the pipeline's inputs.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
        The real ligands followed by every invented one, in acceptance order. The same
        mapping object as the input when generation was off, so the default path pays
        nothing for the stage existing.
    candidates : tuple[Transformation, ...]
        The original pool plus the sub-edges of accepted proposals -- infeasible ones
        included, for the same reason the pipeline keeps every other rejection.
    records : tuple[IntermediateRecord, ...], optional
        One per gap *attempted*, in the order attempted.
    unmet_constraints : tuple[str, ...], optional
        Best-effort budgets generation could not satisfy, phrased for the user and merged
        into the planned network's own list.

    Notes
    -----
    A result object rather than a mutated network because generation runs *before* the
    planner: there is no network yet to mutate, and building one only to replan over it
    would discard the rejections that justified the intermediates in the first place.
    """

    ligands: Mapping[str, Ligand]
    candidates: tuple[Transformation, ...]
    records: tuple[IntermediateRecord, ...] = ()
    unmet_constraints: tuple[str, ...] = ()

    @property
    def synthetic_names(self) -> tuple[str, ...]:
        """Names of the invented vertices, in acceptance order."""
        return tuple(name for name, ligand in self.ligands.items() if ligand.synthetic)


def _dedupe_key(ligand: Ligand) -> tuple[str, int]:
    """Return the identity a synthetic ligand is deduped on: structure and charge.

    Canonical SMILES with hydrogens suppressed, so a molecule built with explicit
    hydrogens and the same molecule without them are one entry, and net formal charge,
    because two protonation states of the same skeleton are genuinely different ligands.
    """
    from rdkit import Chem

    return (Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(ligand.mol))), ligand.charge)


def _intermediate_gaps(
    ligands: Mapping[str, Ligand], candidates: Sequence[Transformation], network_options: NetworkOptions
) -> list[tuple[str, str]]:
    """Return the pairs worth offering to a generator, best first.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
    candidates : Sequence[Transformation]
        The settled pool, feasible and not.
    network_options : NetworkOptions

    Returns
    -------
    list[tuple[str, str]]
        Unordered pairs, ranked by decreasing fingerprint similarity and truncated to
        ``intermediates.max_gaps``.

    Notes
    -----
    **A banned pair is never a gap.** ``A~M~B`` is a way of running exactly the comparison
    the user forbade, at twice the cost, so honouring the ban only on the direct edge
    would honour it in letter and break it in substance.

    Ranking is by fingerprint similarity for the same reason the adaptive loop uses it: it
    costs no mapping, and the pairs most likely to have a bridgeable intermediate are the
    ones already most alike. ``max_gaps`` then cuts the tail, where the pairs are least
    similar and a generator is least likely to find anything.
    """
    options = network_options.intermediates
    names = list(ligands)
    graph = feasible_graph(names, candidates)
    membership = {name: index for index, component in enumerate(nx.connected_components(graph)) for name in component}
    feasible_pairs = {candidate.unordered_key for candidate in candidates if candidate.feasible}

    gaps: set[tuple[str, str]] = set()
    for candidate in candidates:
        pair = candidate.unordered_key
        if pair in feasible_pairs:
            continue
        crosses = membership[pair[0]] != membership[pair[1]]
        if crosses or options.fills_internal_gaps:
            gaps.add(pair)
    # A forced pair with no feasible mapping is offered whatever the mode and whatever the
    # components say: the user demanded that comparison, and an intermediate is the only
    # way to keep it relative.
    for pair in network_options.forced_pairs:
        if pair not in feasible_pairs and pair[0] in ligands and pair[1] in ligands:
            gaps.add(pair)
    gaps -= network_options.banned_pairs

    ranked = sorted(gaps)
    similarities = fingerprint_pair_similarities(ligands, ranked)
    ranked.sort(key=lambda pair: (-similarities[pair], pair))
    if options.max_gaps is not None:
        ranked = ranked[: options.max_gaps]
    return ranked


def _synthesize_molecules(
    proposal: "IntermediateProposal",
    pool: Mapping[str, Ligand],
    generator: "AbstractIntermediateGenerator",
    network_options: NetworkOptions,
    mapping_options: MappingOptions,
    *,
    limit: int,
    known: Mapping[tuple[str, int], str],
    trace: list[str],
) -> dict[str, Ligand]:
    """Pose a proposal's molecules and keep the ones that are new and legal."""
    from rbfenetmap.core.intermediates import synthesize_ligand

    made: dict[str, Ligand] = {}
    for proposed in proposal.molecules:
        if len(made) >= limit:
            trace.append(f"stopped at {limit} molecule(s) for this gap")
            break
        missing = [name for name in proposed.parents if name not in pool]
        if missing:
            trace.append(f"proposal names unknown parent(s) {missing}; skipped")
            continue
        ligand, result = synthesize_ligand(
            proposed,
            {name: pool[name] for name in proposed.parents},
            generator=generator.name,
            softcore=network_options.softcore,
            options=network_options.intermediates,
            mapping_options=mapping_options,
        )
        trace.extend(result.trace)
        if ligand is None:
            continue
        duplicate = known.get(_dedupe_key(ligand))
        if duplicate is not None:
            # Not a failure. The molecule the generator wanted is already a vertex, so the
            # bridge it was meant to build either exists or was already found infeasible.
            trace.append(f"{ligand.name} duplicates {duplicate}; not added again")
            continue
        if ligand.name in pool or ligand.name in made:
            trace.append(f"{ligand.name} was already invented; not added again")
            continue
        made[ligand.name] = ligand
    return made


def _sub_edge_pairs(
    proposal: "IntermediateProposal", made: Mapping[str, Ligand], pool: Mapping[str, Ligand], trace: list[str]
) -> list[tuple[str, str]]:
    """Return the pairs a proposal's links ask to have evaluated.

    A generator that supplies no links is taken to mean the obvious thing -- each new
    molecule against each of its parents -- rather than being treated as having proposed
    nothing. Links naming a molecule that failed posing are dropped with a note, because
    the interesting fact for the user is the pose failure, not its consequence.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    links: list[tuple[str, str]] = [(link.source, link.target) for link in proposal.links]
    if not links:
        links = [(parent, name) for name, ligand in made.items() for parent in ligand.provenance.parents]
    for source, target in links:
        if source == target:
            continue
        if source not in made and target not in made:
            trace.append(f"link {source}~{target} names no surviving molecule; skipped")
            continue
        if (source not in made and source not in pool) or (target not in made and target not in pool):
            trace.append(f"link {source}~{target} names an unknown ligand; skipped")
            continue
        pair = (source, target) if source < target else (target, source)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _connecting_subgraph(
    source: str, target: str, made: Mapping[str, Ligand], sub_edges: Sequence[Transformation]
) -> "nx.Graph | None":
    """Return the part of the sub-edge graph that actually bridges *source* to *target*.

    ``None`` when the feasible sub-edges leave the two ends unconnected, which is the
    signal to drop the proposal whole. Synthetic vertices of degree one are then pruned
    repeatedly: they hang off the bridge without contributing to it, and every one of them
    is a vertex the spanning network would have to pay an edge to reach.

    Only the proposal's *own* sub-edges are in this graph. Asking instead whether the two
    ends are connected in the augmented pool would answer yes for any gap inside an
    already-connected component, which is precisely the case ``mode="gaps"`` exists to
    address and precisely where a useless proposal must still be refused.
    """
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from((source, target))
    for edge in sub_edges:
        if edge.feasible:
            graph.add_edge(*edge.unordered_key)
    if not nx.has_path(graph, source, target):
        return None
    graph = graph.subgraph(nx.node_connected_component(graph, source)).copy()
    pruning = True
    while pruning:
        pruning = False
        for node in [n for n in graph if n in made and graph.degree(n) <= 1]:
            graph.remove_node(node)
            pruning = True
    if not nx.has_path(graph, source, target):  # pragma: no cover - pruning cannot cut the path
        return None
    return graph


def augment_with_intermediates(
    ligands: Mapping[str, Ligand],
    candidates: Sequence[Transformation],
    generator: "AbstractIntermediateGenerator",
    mapper: AbstractMapper,
    scorer: AbstractScorer,
    mapping_options: MappingOptions,
    network_options: NetworkOptions,
) -> AugmentationResult:
    """Invent ligands for the gaps no mapping could cross, and score the new sub-edges.

    The fifth stage, between scoring and planning: **map -> repair -> score -> bridge ->
    plan**.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
        The real ligands.
    candidates : Sequence[Transformation]
        The settled candidate pool, feasible and not. Its rejections are what define the
        gaps.
    generator : AbstractIntermediateGenerator
        Proposes molecules. It never decides feasibility and never prices anything.
    mapper, scorer : AbstractMapper or AbstractScorer
        **The user's own**, unchanged. Every proposed sub-edge is put through
        :func:`build_candidate` with them and with the run's
        :class:`~rbfenetmap.core.options.SoftcorePolicy`.
    mapping_options : MappingOptions
    network_options : NetworkOptions
        Supplies :attr:`~rbfenetmap.core.options.NetworkOptions.intermediates`, the bans,
        the forced pairs, and the ``n_edges`` headroom.

    Returns
    -------
    AugmentationResult
        With the input mapping and pool returned unchanged when
        ``intermediates.mode == "off"``.

    Notes
    -----
    **The generator proposes; the existing feasibility machinery is the sole judge.**
    Nothing here fabricates a :class:`~rbfenetmap.core.models.Transformation` the way
    :func:`~rbfenetmap.core.cbfe.make_cbfe_transformation` legitimately does, and the
    difference is not stylistic: a counterpoised edge has no geometry to check, while an
    intermediate edge is nothing *but* geometry. A badly posed molecule has to come back
    as an ordinary ``core_geometry_mismatch``, which is exactly what routing through
    :func:`build_candidate` makes happen.

    **A proposal is accepted or dropped whole.** If the surviving feasible sub-edges do
    not connect the two ends of the gap, the molecules go too -- there is no such thing as
    a partially useful intermediate, and an orphan synthetic vertex would be a ligand
    nobody can compute a free energy for.

    **The edge budget is spent, not inflated.** See
    :meth:`~rbfenetmap.core.options.NetworkOptions.intermediate_headroom`. Running out is
    recorded on ``unmet_constraints`` rather than raised, because it leaves a perfectly
    valid network.

    **This stage is deliberately serial.** Posing consumes a fixed seed per molecule and
    naming is content-addressed, but the *order* in which gaps consume the shared budget
    is not commutative -- so the pool is evaluated under ``jobs`` while the augmentation
    over it is not, and the output is identical at any ``jobs``.
    """
    options = network_options.intermediates
    if not options.enabled:
        return AugmentationResult(ligands=ligands, candidates=tuple(candidates))

    pool: dict[str, Ligand] = dict(ligands)
    scored: list[Transformation] = list(candidates)
    gaps = _intermediate_gaps(pool, scored, network_options)
    logger.info("Intermediate generation (mode=%r) offered %d gap(s) to %r", options.mode, len(gaps), generator.name)

    records: list[IntermediateRecord] = []
    unmet: list[str] = []
    invented: dict[str, Ligand] = {}
    known: dict[tuple[str, int], str] = {_dedupe_key(ligand): name for name, ligand in pool.items()}
    headroom = network_options.intermediate_headroom(len(pool))
    budget = options.max_intermediates

    for index, (source, target) in enumerate(gaps):
        remaining_gaps = len(gaps) - index
        if headroom is not None and headroom <= 0:
            unmet.append(
                f"n_edges={network_options.n_edges} left no room for intermediates; "
                f"{remaining_gaps} gap(s) were not bridged"
            )
            break
        if budget is not None and budget <= 0:
            unmet.append(
                f"max_intermediates={options.max_intermediates} reached; {remaining_gaps} gap(s) were not bridged"
            )
            break

        caps = [options.max_molecules, *(value for value in (headroom, budget) if value is not None)]
        record = _bridge_one_gap(
            source,
            target,
            pool,
            invented,
            known,
            generator,
            mapper,
            scorer,
            mapping_options,
            network_options,
            scored,
            limit=min(caps),
        )
        records.append(record)
        if record.accepted:
            headroom = None if headroom is None else headroom - len(record.names)
            budget = None if budget is None else budget - len(record.names)

    if invented:
        logger.info("Invented %d ligand(s): %s", len(invented), sorted(invented))
    return AugmentationResult(
        ligands={**pool, **invented}, candidates=tuple(scored), records=tuple(records), unmet_constraints=tuple(unmet)
    )


def _bridge_one_gap(
    source: str,
    target: str,
    pool: Mapping[str, Ligand],
    invented: dict[str, Ligand],
    known: dict[tuple[str, int], str],
    generator: "AbstractIntermediateGenerator",
    mapper: AbstractMapper,
    scorer: AbstractScorer,
    mapping_options: MappingOptions,
    network_options: NetworkOptions,
    scored: list[Transformation],
    *,
    limit: int,
) -> IntermediateRecord:
    """Attempt one gap, mutating *invented*, *known* and *scored* only on acceptance.

    Kept separate from :func:`augment_with_intermediates` so that the all-or-nothing rule
    is enforced by control flow rather than by discipline: every early return here is a
    refusal that has added nothing.
    """

    def refuse(reason: str, trace: Sequence[str]) -> IntermediateRecord:
        """Record an attempt that contributed no ligand."""
        return IntermediateRecord(
            source=source, target=target, generator=generator.name, rejection=reason, trace=tuple(trace)
        )

    available = {**pool, **invented}
    if not generator.supports_pair(available[source], available[target]):
        return refuse("generator_declined_pair", ())

    proposal = generator.propose(available[source], available[target], network_options.intermediates, mapping_options)
    trace: list[str] = list(proposal.trace)
    if not proposal.proposed:
        return refuse(proposal.rejection or "nothing_proposed", trace)

    made = _synthesize_molecules(
        proposal, available, generator, network_options, mapping_options, limit=limit, known=known, trace=trace
    )
    if not made:
        return refuse(proposal.rejection or "no_molecule_survived_posing", trace)

    augmented = {**available, **made}
    sub_edges = [
        build_candidate(augmented[a], augmented[b], mapper, scorer, mapping_options, network_options)
        for a, b in _sub_edge_pairs(proposal, made, available, trace)
    ]
    for edge in sub_edges:
        reasons = ", ".join(reason.value for reason in edge.score.rejections) or "unknown"
        trace.append(f"sub-edge {edge.key}: {'feasible' if edge.feasible else f'rejected ({reasons})'}")

    bridge = _connecting_subgraph(source, target, made, sub_edges)
    if bridge is None:
        trace.append("feasible sub-edges do not connect the gap; proposal dropped whole")
        return refuse("sub_edges_do_not_bridge", trace)

    adopted = {name: ligand for name, ligand in made.items() if name in bridge}
    dropped = sorted(set(made) - set(adopted))
    if dropped:
        trace.append(f"dropped molecule(s) contributing nothing to the bridge: {dropped}")
    if not adopted:
        trace.append("the gap was bridged without any invented molecule; nothing to adopt")
        return refuse("sub_edges_do_not_bridge", trace)
    invented.update(adopted)
    known.update({_dedupe_key(ligand): name for name, ligand in adopted.items()})
    scored.extend(edge for edge in sub_edges if edge.source in bridge and edge.target in bridge)
    return IntermediateRecord(
        source=source, target=target, generator=generator.name, accepted=True, names=tuple(adopted), trace=tuple(trace)
    )


def _plan_over_augmented_pool(
    ligands: Mapping[str, Ligand],
    candidates: Sequence[Transformation],
    mapper: AbstractMapper,
    scorer: AbstractScorer,
    planner: AbstractNetworkPlanner,
    mapping_options: MappingOptions,
    network_options: NetworkOptions,
) -> Network:
    """Run generation over a settled pool, then plan, then attach the record.

    The one place the fourth and fifth stages meet, shared by the eager and adaptive
    paths so they cannot diverge.

    Notes
    -----
    **No precedence logic sits between this and ``cbfe_mode``, and none is needed.** CBFE
    eligibility is evaluated inside the planner against the components of the pool it was
    handed. Generation runs first and changes that pool, so a gap an intermediate closed is
    no longer a gap when the planner computes components and CBFE never triggers for it;
    a gap generation could not close is still a gap, and ``cbfe_mode="bridge"`` still
    rescues it. "Stay relative, fall back to counterpoised, only then fail" falls out of
    stage order, and a flag expressing it would be a second, disagreeable source of truth.

    The generator is constructed **only** when the feature is on, which is also what keeps
    the lazy-import rule: a run that does not ask for intermediates never imports one.

    A disconnection that survives generation carries the attempt record out through the
    refusal, the way ``cmd_plan`` carries its geometry hint out through one. The planner
    is handed a pool, not a history, so it cannot write that paragraph itself -- and a
    user who switched generation on and still got a disconnection needs to know which of
    their gaps were offered and why each was refused.
    """
    result = AugmentationResult(ligands=ligands, candidates=tuple(candidates))
    if network_options.generates_intermediates:
        from rbfenetmap.plugins.intermediates import create_intermediate

        generator = create_intermediate(network_options.intermediates.generator)
        result = augment_with_intermediates(
            ligands, candidates, generator, mapper, scorer, mapping_options, network_options
        )

    try:
        network = planner.plan(result.ligands, result.candidates, network_options)
    except NetworkPlanError as exc:
        from rbfenetmap.core.intermediates import describe_intermediate_attempts

        paragraph = describe_intermediate_attempts(result.records)
        if not paragraph:
            raise
        raise NetworkPlanError(f"{exc}\n{paragraph}", rejected=exc.rejected) from exc

    # Both, not just the records: a budget that left no room stops generation before the
    # first gap is attempted, so there is a message to carry with no record behind it.
    if not result.records and not result.unmet_constraints:
        return network
    return replace(
        network, intermediates=result.records, unmet_constraints=(*network.unmet_constraints, *result.unmet_constraints)
    )


def build_network(
    ligands: Sequence[Ligand] | Mapping[str, Ligand],
    *,
    mapper: AbstractMapper | str = "mcss-e2",
    scorer: AbstractScorer | str = "linear",
    planner: AbstractNetworkPlanner | str = "mst",
    mapping_options: MappingOptions | None = None,
    network_options: NetworkOptions | None = None,
) -> Network:
    """Plan a perturbation network over *ligands*.

    Parameters
    ----------
    ligands : Sequence[Ligand] or Mapping[str, Ligand]
        The vertices. Names must be unique.
    mapper, scorer, planner : AbstractMapper or AbstractScorer or AbstractNetworkPlanner or str
        Plugin instances, or names to look up in the built-in registries. Under
        ``network_options.cbfe_mode == "all"`` the mapper and scorer are never used -- a
        counterpoised edge has no core to map and a closed-form cost -- and a mapper name
        is not even resolved, so an unavailable optional mapper is not an error there.
    mapping_options : MappingOptions, optional
    network_options : NetworkOptions, optional

    Returns
    -------
    Network
        With :attr:`~rbfenetmap.core.models.Network.edges` selected and
        :attr:`~rbfenetmap.core.models.Network.candidates` holding everything scored.

    Raises
    ------
    ValueError
        If fewer than two ligands are supplied, or two share a name.
    rbfenetmap.core.exceptions.NetworkPlanError
        If the user's constraints cannot be satisfied.

    Examples
    --------
    >>> network = build_network(ligands, mapper="cartograph")  # doctest: +SKIP
    >>> len(network.edges)  # doctest: +SKIP
    11
    """
    from rbfenetmap.plugins.mappers import create_mapper
    from rbfenetmap.plugins.planners import create_planner
    from rbfenetmap.plugins.scorers import create_scorer

    if not isinstance(ligands, Mapping):
        indexed: dict[str, Ligand] = {}
        for ligand in ligands:
            if ligand.name in indexed:
                raise ValueError(
                    f"Duplicate ligand name {ligand.name!r}. Names are network vertex identifiers and must be unique."
                )
            indexed[ligand.name] = ligand
        ligands = indexed

    if len(ligands) < 2:
        raise ValueError(f"A network needs at least two ligands; got {len(ligands)}.")

    mapping_options = mapping_options or MappingOptions()
    network_options = network_options or NetworkOptions()
    network_options.check_edge_budget(len(ligands))
    # Reserved only when generation is on. A user with a ligand honestly named ``int_3``
    # is running a plan that cannot invent a colliding name, and refusing it would be a
    # compatibility break bought for nothing.
    reserve_intermediate_names(ligands, enabled=network_options.generates_intermediates)

    scorer_obj = create_scorer(scorer) if isinstance(scorer, str) else scorer
    planner_obj = create_planner(planner) if isinstance(planner, str) else planner
    # Before the mapping stage, so a planner that cannot honour the requested CBFE mode
    # fails in the first second of a run rather than after several minutes of MCS work.
    planner_obj.check_cbfe_support(network_options)
    planner_obj.check_design_support(network_options)

    if network_options.cbfe_mode == "all":
        # No mapper is resolved at all: a counterpoised edge has no common core to find, so
        # every MCS search would be work whose result is discarded. On a large series this
        # is the difference between minutes and milliseconds.
        candidates = _all_cbfe_candidates(ligands, network_options)
        logger.info("cbfe_mode='all': planning over %d counterpoised candidate(s), no mapping", len(candidates))
        # A counterpoised network has no common cores to make consistent, so the gate is a
        # no-op here. It is still called, because a path out of this function that skips it
        # is how the option came to mean nothing on some routes and something on others.
        return maybe_apply_graph_consistency(
            planner_obj.plan(ligands, candidates, network_options), network_options, scorer=scorer_obj
        )

    mapper_obj = create_mapper(mapper) if isinstance(mapper, str) else mapper

    pairs, restored = generate_candidate_pairs(ligands, network_options)
    if restored:
        logger.info(
            "Prefilter would have disconnected the candidate pool; restored %d bridging pair(s): %s",
            len(restored),
            restored,
        )
    logger.info("Evaluating %d candidate pair(s) with mapper %r", len(pairs), mapper_obj.name)

    if network_options.pair_evaluation == "adaptive" and planner_obj.name == "mst":
        # The pool, not the plan: generation runs once after the loop settles, never
        # inside it. Generating mid-loop would satisfy connectivity with synthetic nodes
        # and stop the very RBFE expansion the loop exists to drive -- the same rationale
        # `_adaptive_candidate_pool` already documents for probing with CBFE off.
        candidates = _adaptive_candidate_pool(
            ligands, pairs, mapper_obj, scorer_obj, planner_obj, mapping_options, network_options
        )
    else:
        if network_options.pair_evaluation == "adaptive":
            logger.info("Planner %r requires eager candidate evaluation; mapping the full pool", planner_obj.name)
        candidates = evaluate_pairs(ligands, pairs, mapper_obj, scorer_obj, mapping_options, network_options)
        n_feasible = sum(1 for c in candidates if c.feasible)
        logger.info("%d of %d candidate(s) are feasible", n_feasible, len(candidates))

    # Phase 4b collapsed three exits into one, so the graph-consistency gate that Phase 5
    # had to repeat at each of them is applied here exactly once. That is what Phase 5's
    # docstring asks for -- "a single gate, called on every path out of the pipeline" --
    # now guaranteed by the control flow rather than maintained by hand.
    return maybe_apply_graph_consistency(
        _plan_over_augmented_pool(
            ligands, candidates, mapper_obj, scorer_obj, planner_obj, mapping_options, network_options
        ),
        network_options,
        scorer=scorer_obj,
    )
