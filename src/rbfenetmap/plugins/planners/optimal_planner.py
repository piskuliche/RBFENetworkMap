"""Statistical optimal design: choose edges to minimise a variance criterion.

The default planner asks "what is the cheapest set of edges that connects everything and
closes enough cycles?". This one asks a different question -- "which set of edges, at this
budget, gives the most precise free energies?" -- and answers it with the classical theory
of optimal experimental design, using the fact that
:mod:`rbfenetmap.core.design` derives at length: the Fisher information matrix of a network
of relative measurements **is** its weighted graph Laplacian.

That reframing turns network selection into a subset-selection problem over a matrix
criterion:

``a_optimal``
    Minimise :math:`\\operatorname{tr} C`, the summed variance of the estimates. The right
    choice when each ligand's own number is what matters.

``d_optimal``
    Minimise :math:`\\ln \\det C`, the volume of the joint confidence ellipsoid. Because
    the pseudo-determinant of a Laplacian counts weighted spanning trees, a D-optimal
    design comes out markedly more cyclic than an A-optimal one at the same edge count --
    Pitman reports 40-80% more cycles -- which is why it is the recommendation **when a
    cycle-closure correction will be applied downstream**. Otherwise prefer A-optimal.

Both criteria are lowest-is-best, and both read :attr:`EdgeScore.total
<rbfenetmap.core.models.EdgeScore.total>` as a predicted standard deviation in kcal/mol.
That is what :class:`~rbfenetmap.plugins.scorers.variance_scorer.VarianceScorer` returns;
under any other scorer the planner still runs, on a scale with no physical meaning.

Why a heuristic, and which one
------------------------------
Choosing the best :math:`k`-subset of :math:`\\binom{n}{2}` candidate edges is
combinatorial, and even a 20-ligand series puts it out of reach of enumeration. This ships
**Xu's Appendix-H heuristic** as the default, in three stages:

1. the cheapest spanning tree, so the connectivity guarantee is established before anything
   else competes for the budget;
2. a candidate pool capped at :math:`M = 3n` edges -- the spanning tree plus the cheapest
   remaining candidates;
3. greedy descent on the chosen criterion within that pool, up to the edge budget.

Published as landing within **1.10 ± 0.03x** of the true optimum, and it needs nothing but
numpy and networkx. Measured here against exhaustive enumeration on small complete graphs,
the worst ratio over 40 randomised instances is 1.03x (A-optimal) and 1.08x (D-optimal).

One deviation, and the reason for it
------------------------------------
Appendix H's first stage is the cheapest **2-edge-connected** spanning subgraph, not a
spanning tree. Forcing that turned out to cost more than it buys: choosing a bridge cover by
*cost* spends part of the budget on edges the criterion would rather have spent elsewhere,
and on the same randomised instances it pushes the D-optimal result out to 1.33x of the
optimum where letting the criterion spend that budget itself stays at 1.08x. The greedy
descent removes the bridges that are worth removing anyway -- a bridge has a large effective
resistance, which is exactly what the criterion rewards closing. Any bridge that survives is
recorded on ``unmet_constraints`` rather than bought out.

:class:`OptimalDesignPlanner` also offers **Fedorov exchange** as an opt-in refinement
(``design_refine``): repeatedly swap the in-design edge whose removal costs least for the
out-of-design edge whose addition helps most, until no swap improves the criterion. On the
same instances it brings the worst case to 1.005x. It is written here in numpy on purpose --
HiMap's route to the same answer goes through ``rpy2==3.4.5`` and ``scikit-learn==0.23.2``
and needs an R installation, a dependency footprint out of all proportion to a matrix
criterion over a few hundred edges.

The edge budget
---------------
``n_edges`` still caps selection. When it is unset this planner uses Pitman's floor,
:math:`k_{\\min} = \\operatorname{round}(n \\ln n)`, rather than the package-wide default of
"as many as redundancy wants" -- below that bound precision degrades *worse as n grows*, so
a design planner that ignored it would be optimising within a budget known to be too small.
This is a property of this planner, not a change to the default ``n_edges``: the ``mst``
planner is untouched and ``--compat v0.4`` is unaffected.
"""

from __future__ import annotations

import logging
import math
from typing import ClassVar, Mapping, Sequence

import networkx as nx

from rbfenetmap.core.design import criterion_value, fisher_information
from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.meta.planners import AbstractNetworkPlanner
from rbfenetmap.core.models import EDGE_SEPARATOR, Ligand, Network, Transformation
from rbfenetmap.core.options import NetworkOptions

# Sibling reuse rather than a second copy. These three are the shared vocabulary of
# selection -- collapse a pair to its cheapest orientation, orient a chosen edge, explain a
# disconnection -- and two planners that described a disconnection differently would be a
# worse outcome than reaching across a leading underscore inside one plugin package.
from rbfenetmap.plugins.planners.mst_planner import _best_by_pair, _describe_disconnection, _orient

__all__ = ("OptimalDesignPlanner", "minimum_edge_count")

logger = logging.getLogger(__name__)


def minimum_edge_count(n_ligands: int) -> int:
    """Return Pitman's edge floor, ``round(n ln n)``, clipped to at least ``n - 1``.

    Parameters
    ----------
    n_ligands : int

    Returns
    -------
    int

    Notes
    -----
    Pitman, Hahn, Tresadern and Mobley derive :math:`k_{\\min} \\approx n \\ln n` as the
    point below which added ligands make precision *worse*, not merely no better. At
    :math:`n = 40` that is 148 edges, against the ~40 that ``edges_per_ligand=2`` buys --
    which is the gap this planner exists to make visible.
    """
    if n_ligands < 2:
        return 0
    return max(n_ligands - 1, int(round(n_ligands * math.log(n_ligands))))


class OptimalDesignPlanner(AbstractNetworkPlanner):
    """Select edges by minimising an A- or D-optimality criterion.

    See the module docstring for the criteria, the heuristic, and the budget rule.
    """

    name: ClassVar[str] = "optimal"
    supports_cbfe: ClassVar[bool] = False
    supports_design: ClassVar[bool] = True

    def plan(
        self, ligands: Mapping[str, Ligand], candidates: Sequence[Transformation], options: NetworkOptions
    ) -> Network:
        """Select a statistically optimal network.

        Raises
        ------
        rbfenetmap.core.exceptions.NetworkPlanError
            If ``design`` is ``"none"`` -- this planner has no criterion to optimise and
            will not silently pick one -- if a forced edge is unavailable, or if the
            feasible pool is disconnected while connectivity is required.
        """
        if options.design == "none":
            raise NetworkPlanError(
                "The 'optimal' planner needs a design criterion, but design='none'. Pass "
                "--design a_optimal, or --design d_optimal if a cycle-closure correction will be "
                "applied downstream. There is no default criterion because the two answer different "
                "questions and neither is a safe guess."
            )

        names = sorted(ligands)
        options.check_edge_budget(len(names))
        unmet: list[str] = []

        feasible = {pair: edge for pair, edge in _best_by_pair(candidates).items() if pair not in options.banned_pairs}
        self._check_forced(options, feasible, candidates)

        graph: nx.Graph = nx.Graph()
        graph.add_nodes_from(names)
        for pair, edge in feasible.items():
            graph.add_edge(pair[0], pair[1], weight=max(float(edge.score.total), 1e-9))

        if len(names) > 1 and not nx.is_connected(graph):
            if options.require_connected:
                raise NetworkPlanError(
                    _describe_disconnection(graph, ligands, candidates, cbfe_mode=options.cbfe_mode),
                    rejected=[c for c in candidates if not c.feasible],
                )
            unmet.append(f"network is disconnected ({nx.number_connected_components(graph)} components)")

        target = self._edge_target(graph, options, unmet)
        selected = self._select(graph, names, options, target, unmet)

        self._report_bridges(graph, selected, unmet)
        self._report_degrees(graph, selected, options, unmet)
        logger.info("%s", self.describe_design(graph, names, selected, options))

        edges = tuple(_orient(feasible[pair], ligands, options.edge_direction) for pair in sorted(selected))
        network = Network(
            ligands=ligands,
            edges=edges,
            candidates=tuple(candidates),
            planner=self.name,
            options=options,
            unmet_constraints=tuple(unmet),
        )
        network.validate(require_connected=options.require_connected)
        return network

    def _check_forced(
        self,
        options: NetworkOptions,
        feasible: Mapping[tuple[str, str], Transformation],
        candidates: Sequence[Transformation],
    ) -> None:
        """Raise if any forced edge is unavailable, quoting why.

        The same contract the ``mst`` planner honours: a forced edge bypasses *scoring* but
        not feasibility, and one that cannot be supplied is an error rather than a silent
        omission. Restated here rather than shared because this planner has no CBFE
        fallback to consider, so the check is genuinely shorter.
        """
        by_pair = {c.unordered_key: c for c in candidates}
        problems: list[str] = []
        for pair in sorted(options.forced_pairs):
            if pair in feasible:
                continue
            candidate = by_pair.get(pair)
            spec = f"{pair[0]}{EDGE_SEPARATOR}{pair[1]}"
            if candidate is None:
                problems.append(f"{spec}: never scored (is either ligand in the input?)")
            else:
                reasons = ", ".join(r.value for r in candidate.score.rejections) or "unknown"
                problems.append(f"{spec}: rejected ({reasons})")
        if problems:
            raise NetworkPlanError(
                "Forced edge(s) cannot be used:\n  " + "\n  ".join(problems) + "\nLoosen the soft-core "
                "budget for these pairs, or remove them from forced_edges."
            )

    def _edge_target(self, graph: nx.Graph, options: NetworkOptions, unmet: list[str]) -> int:
        """Resolve how many edges to select, and record it if the pool cannot supply them."""
        available = graph.number_of_edges()
        if options.n_edges is not None:
            target = options.n_edges
        else:
            target = minimum_edge_count(graph.number_of_nodes())
        if target > available:
            unmet.append(
                f"the design asked for {target} edge(s) but only {available} feasible candidate(s) exist; "
                "selecting all of them"
            )
            target = available
        return max(target, 0)

    def _spanning_seed(
        self, graph: nx.Graph, options: NetworkOptions, target: int, unmet: list[str]
    ) -> set[tuple[str, str]]:
        """Stage 1: the cheapest spanning tree, with forced edges pre-seeded.

        Spanning is established before anything else competes for the budget, so the
        connectivity guarantee this package makes everywhere else holds here too: the later
        stages only ever add. Forced edges go in first, so they survive whatever the budget
        does to the rest.

        Cost, not the criterion, chooses these ``n - 1`` edges. That is deliberate and it is
        cheap: with no cycles yet there is little for a variance criterion to discriminate
        on, and a criterion-greedy pass from an empty graph would spend its whole first
        ``n - 1`` steps rediscovering connectivity.
        """
        selected: set[tuple[str, str]] = {pair for pair in options.forced_pairs if graph.has_edge(*pair)}

        forest: nx.Graph = nx.Graph()
        forest.add_nodes_from(graph.nodes)
        forest.add_edges_from(selected)
        for source, other, data in sorted(graph.edges(data=True), key=lambda e: (e[2]["weight"], e[0], e[1])):
            del data
            if not nx.has_path(forest, source, other):
                forest.add_edge(source, other)
                selected.add(tuple(sorted((source, other))))  # type: ignore[arg-type]

        if len(selected) > target:
            unmet.append(f"n_edges={options.n_edges} is below the {len(selected)} edges needed to span the ligands")
        return selected

    def _candidate_pool(
        self, graph: nx.Graph, selected: set[tuple[str, str]], options: NetworkOptions
    ) -> list[tuple[str, str]]:
        """Stage 2: widen the design's search space to ``M = design_candidate_factor * n``.

        Everything already selected, plus the cheapest edges not yet in it. Capping the
        pool is what keeps the greedy descent below :math:`O(n^2)` criterion evaluations per
        added edge on a large series; the cap is generous enough (3n by default) that the
        optimum is inside it in practice, which is the empirical claim behind the
        1.10x bound.
        """
        limit = max(int(round(options.design_candidate_factor * graph.number_of_nodes())), len(selected))
        remaining = sorted(
            (tuple(sorted(pair)) for pair in graph.edges if tuple(sorted(pair)) not in selected),
            key=lambda pair: (graph.edges[pair]["weight"], pair),
        )
        pool = list(selected) + remaining[: max(limit - len(selected), 0)]
        return sorted(pool)  # type: ignore[arg-type]

    def _criterion(
        self, graph: nx.Graph, nodes: Sequence[str], edges: Sequence[tuple[str, str]], criterion: str
    ) -> float:
        """Evaluate the design criterion for an edge set."""
        sigmas = [graph.edges[pair]["weight"] for pair in edges]
        return criterion_value(fisher_information(nodes, list(edges), sigmas), criterion)

    def _select(
        self, graph: nx.Graph, nodes: Sequence[str], options: NetworkOptions, target: int, unmet: list[str]
    ) -> set[tuple[str, str]]:
        """Run the three heuristic stages, then the optional Fedorov refinement."""
        selected = self._spanning_seed(graph, options, target, unmet)
        pool = self._candidate_pool(graph, selected, options)

        # Stage 3: greedy descent. Each step buys the single edge that lowers the criterion
        # most, which is the natural finish to a pool chosen by cost alone -- cost ranks
        # edges individually, and what a design needs is the edge that helps *this* network.
        while len(selected) < target:
            best: tuple[float, tuple[str, str]] | None = None
            for pair in pool:
                if pair in selected:
                    continue
                value = self._criterion(graph, nodes, sorted(selected | {pair}), options.design)
                if best is None or value < best[0]:
                    best = (value, pair)
            if best is None:
                break
            selected.add(best[1])

        if options.design_refine:
            selected = self._fedorov_exchange(graph, nodes, pool, selected, options)
        return selected

    def _report_bridges(self, graph: nx.Graph, selected: set[tuple[str, str]], unmet: list[str]) -> None:
        """Record any bridge left in the design, without buying an edge to remove it.

        A bridge is an edge nothing checks and whose failure disconnects the network, so a
        design that still has one is worth flagging. It is *reported* rather than repaired
        because forcing two-edge-connectivity costs more than it buys: measured against
        exhaustive enumeration on small complete graphs, spending part of the budget on a
        cost-chosen bridge cover pushes the D-optimal result to 1.33x of the optimum, where
        letting the criterion spend that budget itself stays at 1.08x. The criterion
        removes the bridges that are worth removing, on its own, and keeps the ones that
        are not.
        """
        current: nx.Graph = nx.Graph()
        current.add_nodes_from(graph.nodes)
        current.add_edges_from(selected)
        bridges = sorted(tuple(sorted(edge)) for edge in nx.bridges(current))
        if bridges:
            listed = [f"{a}{EDGE_SEPARATOR}{b}" for a, b in bridges[:6]]
            unmet.append(
                f"the design contains {len(bridges)} bridge(s) -- edge(s) no cycle checks: "
                f"{listed}{'...' if len(bridges) > 6 else ''}; raise n_edges to let the criterion "
                "buy its way out of them"
            )

    def _fedorov_exchange(
        self,
        graph: nx.Graph,
        nodes: Sequence[str],
        pool: Sequence[tuple[str, str]],
        selected: set[tuple[str, str]],
        options: NetworkOptions,
        *,
        max_rounds: int = 50,
    ) -> set[tuple[str, str]]:
        """Refine a design by exchanging one edge at a time.

        Each round scores every (drop, add) pair drawn from the design and the pool and
        applies the single best exchange, stopping as soon as none improves the criterion.
        Forced edges are never dropped, and a swap that would disconnect the network scores
        as infinite, so connectivity survives without a separate guard.

        Opt-in rather than default: it costs :math:`O(k \\cdot |pool|)` criterion
        evaluations per round against the heuristic's :math:`O(|pool|)` per added edge, and
        the heuristic is already within a published 1.10x of the optimum. The refinement
        earns its keep on small, awkward pools -- not on the everyday congeneric series.
        """
        forced = {pair for pair in options.forced_pairs if graph.has_edge(*pair)}
        current = self._criterion(graph, nodes, sorted(selected), options.design)
        for _ in range(max_rounds):
            best: tuple[float, tuple[str, str], tuple[str, str]] | None = None
            for drop in sorted(selected - forced):
                remainder = selected - {drop}
                for add in pool:
                    if add in selected:
                        continue
                    value = self._criterion(graph, nodes, sorted(remainder | {add}), options.design)
                    if value < current - 1e-12 and (best is None or value < best[0]):
                        best = (value, drop, add)
            if best is None:
                break
            current = best[0]
            selected = (selected - {best[1]}) | {best[2]}
        return selected

    def _report_degrees(
        self, graph: nx.Graph, selected: set[tuple[str, str]], options: NetworkOptions, unmet: list[str]
    ) -> None:
        """Record, without acting on, an ``edges_per_ligand`` shortfall.

        The criterion, not the degree target, is what this planner optimises -- forcing a
        degree would be spending budget against the objective the user asked for. But a
        shortfall is still worth knowing about, so it lands on ``unmet_constraints`` exactly
        as it would from the ``mst`` planner.
        """
        degrees = dict.fromkeys(graph.nodes, 0)
        for source, target in selected:
            degrees[source] += 1
            degrees[target] += 1
        deficient = sorted(name for name, degree in degrees.items() if degree < options.edges_per_ligand)
        if deficient:
            unmet.append(
                f"edges_per_ligand={options.edges_per_ligand} unmet for {len(deficient)} ligand(s): "
                f"{deficient[:6]}{'...' if len(deficient) > 6 else ''}; the design criterion, not the "
                "degree target, decides selection here"
            )

    def describe_design(
        self, graph: nx.Graph, nodes: Sequence[str], selected: set[tuple[str, str]], options: NetworkOptions
    ) -> str:
        """Return the one-line design summary recorded on every planned network.

        Both criteria are reported regardless of which was optimised, because the interesting
        comparison is almost always between them -- and because the number is meaningless
        without knowing it came from predicted rather than measured variances.
        """
        edges = sorted(selected)
        sigmas = [graph.edges[pair]["weight"] for pair in edges]
        fisher = fisher_information(nodes, edges, sigmas)
        trace = criterion_value(fisher, "a_optimal")
        logdet = criterion_value(fisher, "d_optimal")
        return (
            f"design={options.design}: {len(edges)} edge(s), predicted tr(C)={trace:.4g}, "
            f"ln det(C)={logdet:.4g} (predicted variances, not measured; optimal design buys "
            "precision, not accuracy)"
        )
