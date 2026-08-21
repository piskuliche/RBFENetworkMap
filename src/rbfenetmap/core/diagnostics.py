"""Network-level metrics over an already-planned network.

Everything here is read-only and pure: a :class:`~rbfenetmap.core.models.Network` goes in,
numbers come out, and nothing is ever selected, rejected, or re-planned. That is what makes
it safe to call from a report renderer and from the CLI without either of them being able
to change what the other sees.

The distinction from ``rbfenet inspect`` is the unit of analysis. ``inspect`` answers
questions about *one edge* -- its mapping, its soft-core, why it was rejected.
``diagnose`` answers questions about *the network*: how long the longest comparison path
is, how much of it survives a failed edge, whether the edge budget is anywhere near the
statistical floor. A reviewer's second question about a planned network, after "why isn't X
connected to Y", is always one of these, and until now the HTML report was a picture with
no numbers beside it.

The metric set follows Konnektor's, with two deliberate departures.

**Every seed is mandatory.** :func:`failure_robustness` is the only Monte-Carlo function in
the package, and ``tests/test_softcore.py`` already asserts that planning is deterministic.
A defaulted seed is a defaulted seed until someone forgets it, so it is a required
argument: an unseeded run is not possible to write by accident.

**The n ln n edge-budget floor is advice, not a warning.** Pitman *et al.*, *JCIM* 2023, 63,
1776-1793 derive ``k_min ~ n ln n`` edges, below which precision degrades *worse as n
grows* -- at 40 ligands that is 148 edges where the default ``edges_per_ligand=2`` buys
about 40. Routing that through :func:`warnings.warn` would fire it on essentially every run
this package has ever planned, and a warning that always fires is a warning nobody reads.
It belongs in a report the user asked for, which is :func:`edge_budget_advice`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rbfenetmap.core.models import Network

__all__ = (
    "DegreeSummary",
    "EdgeBudgetAdvice",
    "FailureRobustness",
    "count_cycles",
    "degree_summary",
    "diameter",
    "edge_budget_advice",
    "failure_robustness",
    "network_cost",
    "network_efficiency",
    "summarize",
)


def _graph(network: "Network") -> nx.Graph:
    """Selected edges as a plain undirected graph over every ligand.

    Isolated ligands are kept as nodes deliberately: a ligand nothing reaches is exactly
    what several of these metrics exist to surface, and dropping it would make the network
    look healthier than it is.
    """
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(network.ligands)
    graph.add_edges_from(edge.unordered_key for edge in network.edges)
    return graph


def network_cost(network: "Network") -> float:
    """Total cost of the selected edges, on the scorer's scale.

    Parameters
    ----------
    network : Network

    Returns
    -------
    float
    """
    return float(sum(edge.score.total for edge in network.edges))


def network_efficiency(network: "Network") -> float:
    """Mean cost per selected edge, on the scorer's scale.

    Parameters
    ----------
    network : Network

    Returns
    -------
    float
        ``0.0`` for a network with no edges, rather than a division error: an empty
        network is a legitimate thing to hand a report renderer.

    Notes
    -----
    Useful only *between* networks over the same ligands. Comparing it across ligand sets
    compares two arbitrary difficulty scales, and comparing it against
    :func:`network_cost` compares a mean against a sum -- a denser network is expected to
    have the higher total and the lower mean at the same time.
    """
    if not network.edges:
        return 0.0
    return network_cost(network) / len(network.edges)


def count_cycles(network: "Network", max_length: int = 4) -> int:
    """Count the simple cycles of at most *max_length* ligands.

    Parameters
    ----------
    network : Network
    max_length : int, optional
        Longest cycle counted. The default of 4 follows cinnabar's convention.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If *max_length* is below three, which no cycle can be.

    Notes
    -----
    The bound is not a performance nicety, it is what makes the function terminate in
    useful time: the simple cycles of a dense graph are exponential in the node count, and
    a redundant network over fifty ligands has enough of them to hang a report. Short
    cycles are also the ones that matter -- a cycle-closure residual over twenty edges
    localises nothing.
    """
    if max_length < 3:
        raise ValueError(f"max_length must be at least 3; a cycle cannot be shorter. Got {max_length}.")
    return sum(1 for _ in nx.simple_cycles(_graph(network), length_bound=max_length))


@dataclass(frozen=True)
class DegreeSummary:
    """Per-ligand edge counts and their extremes.

    Parameters
    ----------
    degrees : dict[str, int]
        Ligand name to number of selected edges touching it.
    minimum, maximum : int
        The extremes. ``0`` for a network with no ligands.
    mean : float
        Mean degree, which is ``2 * n_edges / n_ligands``.
    """

    degrees: dict[str, int]
    minimum: int
    maximum: int
    mean: float

    @property
    def isolated(self) -> tuple[str, ...]:
        """Ligands no selected edge touches, sorted.

        The single most actionable line in a diagnostic report: a ligand at degree zero
        has no measured free energy at all, whatever the rest of the network looks like.
        """
        return tuple(sorted(name for name, degree in self.degrees.items() if degree == 0))


def degree_summary(network: "Network") -> DegreeSummary:
    """Summarize how many edges each ligand carries.

    Parameters
    ----------
    network : Network

    Returns
    -------
    DegreeSummary
    """
    degrees = {name: int(degree) for name, degree in _graph(network).degree()}
    if not degrees:
        return DegreeSummary(degrees={}, minimum=0, maximum=0, mean=0.0)
    values = list(degrees.values())
    return DegreeSummary(degrees=degrees, minimum=min(values), maximum=max(values), mean=sum(values) / len(values))


def diameter(network: "Network") -> int | None:
    """Longest shortest path between two ligands, in edges.

    Parameters
    ----------
    network : Network

    Returns
    -------
    int or None
        ``None`` when the network is disconnected, because the diameter is then infinite
        rather than large and reporting a number for it would be a lie. ``0`` for a single
        ligand.

    Notes
    -----
    Computed with ``usebounds=True``: the bounded form (the FastLomap optimisation,
    arXiv:2304.04713) replaces the all-pairs sweep with a handful of BFS runs, which is
    what keeps this affordable in a report over a hundred ligands.
    """
    graph = _graph(network)
    if graph.number_of_nodes() < 2:
        return 0
    if not nx.is_connected(graph):
        return None
    return int(nx.diameter(graph, usebounds=True))


@dataclass(frozen=True)
class FailureRobustness:
    """What survives when edges fail, measured by Monte-Carlo removal.

    Parameters
    ----------
    connected_fraction : float
        Fraction of trials in which the surviving network still spans every ligand.
    mean_ligands_retained : float
        Mean size of the largest surviving connected component, in ligands. The natural
        companion to the fraction above: a network that stays connected 40% of the time
        but keeps 95% of its ligands the rest of the time is in a very different position
        from one that shatters.
    failure_rate : float
        The per-edge failure probability the trials used.
    n_repeats : int
        How many trials were run.
    seed : int
        The seed they were run with, carried so a reported figure can be reproduced
        without going back to the command line that produced it.
    """

    connected_fraction: float
    mean_ligands_retained: float
    failure_rate: float
    n_repeats: int
    seed: int


def failure_robustness(
    network: "Network", *, failure_rate: float = 0.05, n_repeats: int = 100, seed: int
) -> FailureRobustness:
    """Estimate how much of *network* survives independent edge failures.

    An alchemical edge fails for reasons a planner cannot see -- a sampling problem, a
    crashed run, a pose that turns out wrong. This asks what the network looks like
    afterwards: remove each edge independently with probability *failure_rate*, and see
    whether the rest still hangs together.

    Parameters
    ----------
    network : Network
    failure_rate : float, optional
        Independent per-edge failure probability, in [0, 1].
    n_repeats : int, optional
        Number of Monte-Carlo trials.
    seed : int
        **Required, not optional.** This is the only stochastic function in the package,
        and everything around it asserts determinism. A default here would be a default
        right up until someone left it off, and a diagnostic number that changes between
        two runs of the same command is worse than no number.

    Returns
    -------
    FailureRobustness

    Raises
    ------
    ValueError
        If *failure_rate* is outside [0, 1] or *n_repeats* is not positive.

    Notes
    -----
    Edge failures are treated as independent, which is optimistic: in practice the ligand
    that breaks one edge tends to break its neighbours too, so read the result as an upper
    bound on robustness rather than an estimate of it.
    """
    if not 0.0 <= failure_rate <= 1.0:
        raise ValueError(f"failure_rate must lie in [0, 1]; got {failure_rate}.")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1; got {n_repeats}.")

    graph = _graph(network)
    nodes = list(graph.nodes)
    edges = [tuple(sorted(pair)) for pair in graph.edges]
    if not nodes:
        return FailureRobustness(1.0, 0.0, failure_rate, n_repeats, seed)

    rng = random.Random(seed)
    connected = 0
    retained = 0
    for _ in range(n_repeats):
        survivors = [pair for pair in edges if rng.random() >= failure_rate]
        trial: nx.Graph = nx.Graph()
        trial.add_nodes_from(nodes)
        trial.add_edges_from(survivors)
        largest = max((len(component) for component in nx.connected_components(trial)), default=0)
        retained += largest
        if largest == len(nodes):
            connected += 1
    return FailureRobustness(
        connected_fraction=connected / n_repeats,
        mean_ligands_retained=retained / n_repeats,
        failure_rate=failure_rate,
        n_repeats=n_repeats,
        seed=seed,
    )


@dataclass(frozen=True)
class EdgeBudgetAdvice:
    """How the planned edge count compares with the published precision floor.

    Parameters
    ----------
    n_ligands, n_edges : int
        What was planned.
    recommended : int
        ``ceil(n * ln n)``, the floor Pitman 2023 derives.
    shortfall : int
        ``recommended - n_edges``, clamped at zero.

    Notes
    -----
    Advisory, and deliberately not a warning -- see this module's docstring. The floor is
    also a floor for *precision*, not for correctness: a network below it is a perfectly
    valid network whose free energies simply carry more statistical uncertainty than a
    denser one over the same ligands would, and the gap widens as the series grows.
    """

    n_ligands: int
    n_edges: int
    recommended: int
    shortfall: int

    @property
    def message(self) -> str:
        """One line stating the comparison, suitable for a report or the plan summary."""
        if self.n_ligands < 2:
            return f"{self.n_ligands} ligand(s): no meaningful edge budget."
        if self.shortfall <= 0:
            return (
                f"{self.n_edges} edges over {self.n_ligands} ligands meets the n*ln(n) precision "
                f"floor of {self.recommended}."
            )
        return (
            f"{self.n_edges} edges over {self.n_ligands} ligands is {self.shortfall} below the "
            f"n*ln(n) precision floor of {self.recommended} (Pitman 2023). Below that floor precision "
            "degrades faster as the series grows; raise --edges-per-ligand or --n-edges to buy it back."
        )


def edge_budget_advice(n_ligands: int, n_edges: int) -> EdgeBudgetAdvice:
    """Compare an edge count against the ``n ln n`` precision floor.

    Parameters
    ----------
    n_ligands, n_edges : int

    Returns
    -------
    EdgeBudgetAdvice

    Notes
    -----
    Takes two integers rather than a :class:`~rbfenetmap.core.models.Network` on purpose:
    the most useful moment to ask this is *before* planning, when the only thing that
    exists is a ligand count and a budget.
    """
    if n_ligands < 2:
        return EdgeBudgetAdvice(n_ligands=n_ligands, n_edges=n_edges, recommended=0, shortfall=0)
    recommended = math.ceil(n_ligands * math.log(n_ligands))
    return EdgeBudgetAdvice(
        n_ligands=n_ligands, n_edges=n_edges, recommended=recommended, shortfall=max(0, recommended - n_edges)
    )


def summarize(
    network: "Network", *, seed: int = 0, failure_rate: float = 0.05, n_repeats: int = 100, max_cycle_length: int = 4
) -> dict[str, Any]:
    """Run every diagnostic over *network* and return the results together.

    Parameters
    ----------
    network : Network
    seed : int, optional
        Passed to :func:`failure_robustness`. Defaulted *here* and nowhere else: this is
        the presentation layer, where every call is one of many and the caller is asking
        for a report rather than for a number, so a stable default is what keeps two runs
        of ``rbfenet diagnose`` on the same file agreeing with each other.
    failure_rate, n_repeats : float, int, optional
        Passed to :func:`failure_robustness`.
    max_cycle_length : int, optional
        Passed to :func:`count_cycles`.

    Returns
    -------
    dict[str, Any]
        Keys ``n_ligands``, ``n_edges``, ``n_rbfe``, ``n_cbfe``, ``cost``, ``efficiency``,
        ``n_cycles``, ``max_cycle_length``, ``degrees`` (a :class:`DegreeSummary`),
        ``diameter``, ``robustness`` (a :class:`FailureRobustness`), and ``budget`` (an
        :class:`EdgeBudgetAdvice`).
    """
    return {
        "n_ligands": len(network.ligands),
        "n_edges": len(network.edges),
        "n_rbfe": len(network.rbfe_edges),
        "n_cbfe": len(network.cbfe_edges),
        "cost": network_cost(network),
        "efficiency": network_efficiency(network),
        "n_cycles": count_cycles(network, max_length=max_cycle_length),
        "max_cycle_length": max_cycle_length,
        "degrees": degree_summary(network),
        "diameter": diameter(network),
        "robustness": failure_robustness(network, failure_rate=failure_rate, n_repeats=n_repeats, seed=seed),
        "budget": edge_budget_advice(len(network.ligands), len(network.edges)),
    }
