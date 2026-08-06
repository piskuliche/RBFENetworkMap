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
from concurrent.futures import ProcessPoolExecutor
from typing import Mapping, Sequence

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
from rbfenetmap.core.pairs import generate_candidate_pairs
from rbfenetmap.core.softcore import precheck_mapping, repair_softcore_connectivity

__all__ = ("build_candidate", "build_network", "evaluate_pairs")

logger = logging.getLogger(__name__)


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


def _evaluate_one(args: tuple) -> Transformation:  # pragma: no cover - process-pool entry
    """Top-level worker so the pair evaluation can be pickled for a process pool."""
    return build_candidate(*args)


def evaluate_pairs(
    ligands: Mapping[str, Ligand],
    pairs: Sequence[tuple[str, str]],
    mapper: AbstractMapper,
    scorer: AbstractScorer,
    mapping_options: MappingOptions,
    network_options: NetworkOptions,
) -> list[Transformation]:
    """Map, repair, and score every pair.

    Parallelised over ``network_options.jobs``. Pairs are independent, so this is a plain
    fan-out with no shared state.
    """
    work = [
        (ligands[source], ligands[target], mapper, scorer, mapping_options, network_options) for source, target in pairs
    ]
    if network_options.jobs > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=network_options.jobs) as pool:
            return list(pool.map(_evaluate_one, work))
    return [build_candidate(*item) for item in work]


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
        Plugin instances, or names to look up in the built-in registries.
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

    mapper_obj = create_mapper(mapper) if isinstance(mapper, str) else mapper
    scorer_obj = create_scorer(scorer) if isinstance(scorer, str) else scorer
    planner_obj = create_planner(planner) if isinstance(planner, str) else planner

    pairs, restored = generate_candidate_pairs(ligands, network_options)
    if restored:
        logger.info(
            "Prefilter would have disconnected the candidate pool; restored %d bridging pair(s): %s",
            len(restored),
            restored,
        )
    logger.info("Evaluating %d candidate pair(s) with mapper %r", len(pairs), mapper_obj.name)

    candidates = evaluate_pairs(ligands, pairs, mapper_obj, scorer_obj, mapping_options, network_options)
    n_feasible = sum(1 for c in candidates if c.feasible)
    logger.info("%d of %d candidate(s) are feasible", n_feasible, len(candidates))

    return planner_obj.plan(ligands, candidates, network_options)
