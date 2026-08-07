"""The four-stage pipeline: map, repair, score, plan.

:func:`build_network` is the package's main entry point. Everything the CLI does, and
everything an embedding program needs, goes through here.

The stage that most shapes the result is the second one. A mapper is allowed to return a
fragmented soft-core; the repair either fixes it or rejects the edge. Rejection is a
normal outcome recorded on the candidate, never an exception -- one impossible pair among
several hundred must not abort a run.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Callable, Mapping, Sequence

import networkx as nx

from rbfenetmap.core.cbfe import build_cbfe_pool, make_cbfe_transformation
from rbfenetmap.core.descriptors import compute_descriptors
from rbfenetmap.core.exceptions import MappingError, RepairError
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.meta.planners import AbstractNetworkPlanner
from rbfenetmap.core.meta.scorers import AbstractScorer
from rbfenetmap.core.models import (
    AtomMapping,
    EdgeScore,
    Ligand,
    Network,
    RejectionReason,
    SoftcoreRepair,
    Transformation,
)
from rbfenetmap.core.options import MappingOptions, NetworkOptions
from rbfenetmap.core.pairs import fingerprint_pair_similarities, generate_candidate_pairs
from rbfenetmap.core.softcore import precheck_mapping, repair_softcore_connectivity

__all__ = ("build_candidate", "build_network", "evaluate_pairs", "evaluate_pairs_adaptively")

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


def _feasible_graph(names: Sequence[str], candidates: Sequence[Transformation]) -> nx.Graph:
    """Build the undirected graph of candidates that passed feasibility checks."""
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


def evaluate_pairs_adaptively(
    ligands: Mapping[str, Ligand],
    pairs: Sequence[tuple[str, str]],
    mapper: AbstractMapper,
    scorer: AbstractScorer,
    planner: AbstractNetworkPlanner,
    mapping_options: MappingOptions,
    network_options: NetworkOptions,
) -> Network:
    """Evaluate promising pairs in batches until the network targets are met.

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
            feasible_graph = _feasible_graph(names, candidates)
            components = list(nx.connected_components(feasible_graph))
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
    # Re-plan outside warning suppression so any genuinely unmet best-effort target is
    # visible exactly once. For a required but impossible connection this raises with
    # diagnostics after all component-bridging possibilities have been attempted.
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

    scorer_obj = create_scorer(scorer) if isinstance(scorer, str) else scorer
    planner_obj = create_planner(planner) if isinstance(planner, str) else planner
    # Before the mapping stage, so a planner that cannot honour the requested CBFE mode
    # fails in the first second of a run rather than after several minutes of MCS work.
    planner_obj.check_cbfe_support(network_options)

    if network_options.cbfe_mode == "all":
        # No mapper is resolved at all: a counterpoised edge has no common core to find, so
        # every MCS search would be work whose result is discarded. On a large series this
        # is the difference between minutes and milliseconds.
        candidates = _all_cbfe_candidates(ligands, network_options)
        logger.info("cbfe_mode='all': planning over %d counterpoised candidate(s), no mapping", len(candidates))
        return planner_obj.plan(ligands, candidates, network_options)

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
        return evaluate_pairs_adaptively(
            ligands, pairs, mapper_obj, scorer_obj, planner_obj, mapping_options, network_options
        )
    if network_options.pair_evaluation == "adaptive":
        logger.info("Planner %r requires eager candidate evaluation; mapping the full pool", planner_obj.name)

    candidates = evaluate_pairs(ligands, pairs, mapper_obj, scorer_obj, mapping_options, network_options)
    n_feasible = sum(1 for c in candidates if c.feasible)
    logger.info("%d of %d candidate(s) are feasible", n_feasible, len(candidates))

    return planner_obj.plan(ligands, candidates, network_options)
