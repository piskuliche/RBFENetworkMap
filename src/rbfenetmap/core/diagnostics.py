"""Diagnostics-driven replanning: prune the edges the analysis distrusts, refill the gaps.

An RBFE campaign is a loop -- plan, run, analyse, replan -- and this module is the return
leg. It takes a per-edge diagnostic from the analysis stage, drops the edges that diagnostic
condemns, and hands the pruned pool back to the planner so the network is rebuilt around
what is left.

The Lagrange Multiplier Index
-----------------------------
FE-ToolKit's ``edgembar`` fits the whole network at once under the constraint that every
cycle closes. Each edge's Lagrange multiplier measures how hard that constraint had to pull
on it: a large **Lagrange Multiplier Index (LMI)** means the edge disagrees with the
consensus its cycles impose, which is the signature of a poorly converged or badly set up
transformation. The CBFE paper does exactly this by hand on BACE1 and BRD4 -- inspect the
worst edges, drop them, re-run.

What LMI pruning is and is not worth
------------------------------------
Pruning high-LMI edges **substantially reduces cycle-closure error and leaves MUE and RMSE
against experiment essentially unchanged.** That is not a disappointing result, it is the
correct interpretation of the quantity: hysteresis is a *sampling* diagnostic, not an
accuracy predictor. A network can close every cycle perfectly and still sit a kcal/mol off
the experimental values, because a systematic error in the force field or the protonation
state moves every edge in a cycle the same way and cancels exactly where hysteresis would
have shown it.

So use this to find edges that are internally inconsistent and worth re-running or
replacing. Do not use it, and do not report it, as a route to better agreement with
experiment.

Ingesting the diagnostic
------------------------
:func:`load_edge_lmi` reads a small, explicit format of this package's own -- a mapping of
``"source~target"`` to a number, as a dict or a JSON file. **It does not read edgembar's
on-disk output.** Writing a parser against a format that could not be verified against a
real file would be guesswork dressed up as an integration; extracting the multipliers from
an ``edgembar`` analysis and writing this JSON is a short script on the user's side today,
and a first-class reader is a follow-up that needs a real file to develop against.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.models import EDGE_SEPARATOR, Network, parse_edge_key
from rbfenetmap.core.options import NetworkOptions

__all__ = (
    "cycle_closure_errors",
    "load_edge_lmi",
    "lmi_threshold",
    "replan_after_diagnostics",
    "select_high_lmi_edges",
)

logger = logging.getLogger(__name__)


def load_edge_lmi(
    source: Mapping[str, float] | Mapping[tuple[str, str], float] | str | Path,
) -> dict[tuple[str, str], float]:
    """Read per-edge Lagrange Multiplier Indices into a keyed mapping.

    Parameters
    ----------
    source : Mapping or str or pathlib.Path
        Either a mapping already in memory, or the path to a JSON file. Accepted shapes:

        - ``{"lig_a~lig_b": 0.42, ...}`` -- the usual one;
        - ``{("lig_a", "lig_b"): 0.42, ...}`` -- in memory only, since JSON has no tuple
          keys;
        - ``{"edges": {"lig_a~lig_b": 0.42, ...}}`` -- a wrapper, so a file may carry other
          analysis output alongside.

    Returns
    -------
    dict[tuple[str, str], float]
        Keyed by *unordered* endpoint pair, because an LMI is a property of the
        transformation and the transformation is undirected: an analysis that reports
        ``b~a`` describes the edge the planner selected as ``a~b``.

    Raises
    ------
    ValueError
        If the document is not one of the shapes above, a key is not an edge, a value is not
        a number, or the same unordered pair appears twice with different values -- which
        means the analysis and the network disagree about what the edges are, and picking
        one silently is the failure this package refuses everywhere else.

    Notes
    -----
    This is deliberately a small format of this package's own, not edgembar's. See the
    module docstring: a parser written against a file format nobody could check would be a
    guess presented as an integration.
    """
    if isinstance(source, (str, Path)):
        document = json.loads(Path(source).read_text())
        if not isinstance(document, Mapping):
            raise ValueError(f"{source}: expected a JSON object mapping edge keys to LMI values.")
        raw: Mapping = document.get("edges", document) if "edges" in document else document
    else:
        raw = source

    values: dict[tuple[str, str], float] = {}
    for key, value in raw.items():
        if isinstance(key, str):
            endpoints = parse_edge_key(key)
        else:
            endpoints = tuple(key)  # type: ignore[assignment]
            if len(endpoints) != 2:
                raise ValueError(f"Edge key {key!r} is not a pair of ligand names.")
        pair: tuple[str, str] = tuple(sorted(endpoints))  # type: ignore[assignment]
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"LMI for edge {key!r} is {value!r}, which is not a number.") from exc
        if pair in values and values[pair] != number:
            raise ValueError(
                f"Edge {pair[0]}{EDGE_SEPARATOR}{pair[1]} appears twice with different LMI values "
                f"({values[pair]} and {number}). An LMI is a property of the undirected transformation, "
                "so the two orientations cannot disagree."
            )
        values[pair] = number
    return values


def lmi_threshold(values: Sequence[float], *, quantile: float = 0.9) -> float:
    """Return the *quantile* cut point of *values*, by nearest-rank.

    Parameters
    ----------
    values : Sequence[float]
    quantile : float, optional
        In ``[0, 1]``. ``0.9`` cuts at the worst tenth.

    Returns
    -------
    float
        The smallest value at or above the requested rank. Edges strictly above it are the
        ones pruned, so a network whose LMIs are all equal loses none of them.

    Raises
    ------
    ValueError
        If *values* is empty or *quantile* lies outside ``[0, 1]``.

    Notes
    -----
    Nearest-rank rather than an interpolated quantile, so the threshold is always a value
    the analysis actually reported. An interpolated cut sitting between two edges' LMIs is
    harder to explain to the person who has to justify dropping one of them.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must lie in [0, 1]; got {quantile}.")
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute an LMI threshold from an empty set of values.")
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return ordered[index]


def select_high_lmi_edges(
    network: Network,
    lmi: Mapping[tuple[str, str], float],
    *,
    threshold: float | None = None,
    quantile: float = 0.9,
    max_pruned: int | None = None,
    require_complete: bool = True,
) -> tuple[tuple[str, str], ...]:
    """Return the selected edges whose LMI exceeds the cut, worst first.

    Parameters
    ----------
    network : Network
        Only its *selected* edges are considered; a candidate that was never run has no
        diagnostic to read.
    lmi : Mapping[tuple[str, str], float]
        From :func:`load_edge_lmi`.
    threshold : float, optional
        Absolute cut. Edges with an LMI **strictly greater** than this are selected. When
        omitted the cut is taken from *quantile*.
    quantile : float, optional
        Used only when *threshold* is ``None``.
    max_pruned : int, optional
        Keep at most this many, taking the worst. A guard for the case where the whole
        network scores badly: pruning half of it is a statement that the run failed, not a
        repair, and it should be a deliberate act rather than a quantile's side effect.
    require_complete : bool, optional
        Raise if any selected edge has no LMI value. On by default: treating a missing value
        as zero would silently exempt exactly the edges an analysis failed to produce a
        number for, which are not the edges one wants to trust by default.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Unordered pairs, ordered by descending LMI.

    Raises
    ------
    ValueError
        If *require_complete* is set and an edge is missing a value, or if no selected edge
        has one at all.

    Notes
    -----
    Forced edges are never returned. A user who pinned an edge has asserted it must be in
    the network, and a diagnostic does not override that -- it is reported as skipped
    instead, since "your forced edge is the worst edge in the network" is worth reading.
    """
    selected = [edge.unordered_key for edge in network.edges]
    known = {pair: lmi[pair] for pair in selected if pair in lmi}
    if not known:
        raise ValueError(
            "None of the selected edges has an LMI value. Check that the analysis names ligands the same "
            f"way the network does; the network's edges are {[f'{a}{EDGE_SEPARATOR}{b}' for a, b in selected[:6]]}."
        )
    missing = sorted(pair for pair in selected if pair not in lmi)
    if missing and require_complete:
        raise ValueError(
            f"{len(missing)} selected edge(s) have no LMI value: "
            f"{[f'{a}{EDGE_SEPARATOR}{b}' for a, b in missing[:6]]}. Treating a missing diagnostic as a good "
            "one would exempt exactly the edges the analysis could not produce a number for. Supply their "
            "values, or pass require_complete=False to consider them beyond reproach deliberately."
        )

    cut = threshold if threshold is not None else lmi_threshold(list(known.values()), quantile=quantile)
    forced = _options(network).forced_pairs
    ranked = sorted((pair for pair, value in known.items() if value > cut), key=lambda p: (-known[p], p))
    kept = [pair for pair in ranked if pair not in forced]
    if len(kept) != len(ranked):
        logger.info("Not pruning forced edge(s) despite a high LMI: %s", sorted(set(ranked) - set(kept)))
    if max_pruned is not None:
        kept = kept[:max_pruned]
    logger.info("LMI cut at %.4g selects %d of %d selected edge(s)", cut, len(kept), len(selected))
    return tuple(kept)


def _options(network: Network) -> NetworkOptions:
    """Return the network's options, or defaults for a network planned without any."""
    return network.options if network.options is not None else NetworkOptions()


def replan_after_diagnostics(
    network: Network,
    lmi: Mapping[tuple[str, str], float],
    *,
    threshold: float | None = None,
    quantile: float = 0.9,
    max_pruned: int | None = None,
    require_complete: bool = True,
    keep_existing: bool = True,
    planner: str = "mst",
) -> tuple[Network, tuple[tuple[str, str], ...]]:
    """Prune the high-LMI edges and re-plan the network without them.

    Parameters
    ----------
    network : Network
        A network whose candidate pool is intact -- ``rbfenet plan`` writes it, and it is
        what makes this cheap. Not modified.
    lmi : Mapping[tuple[str, str], float]
    threshold, quantile, max_pruned, require_complete
        Passed to :func:`select_high_lmi_edges`.
    keep_existing : bool, optional
        Hold the surviving edges of the current network in place, so the replan changes
        only the gaps. On by default, and it is the difference between a replan you can act
        on and one you cannot: those edges are set up, queued, or already finished, and a
        selection pass free to reshuffle them hands back a network that is not the one being
        run. Turn it off for a clean re-selection over the pruned pool -- the right choice
        before anything has been submitted.
    planner : str, optional
        Planner plugin used for the re-plan. The default re-runs the one this package
        plans with.

    Returns
    -------
    tuple[Network, tuple[tuple[str, str], ...]]
        The replanned network and the pairs that were pruned. The pruned pairs are also on
        the returned network's ``options.banned_edges``.

    Raises
    ------
    rbfenetmap.core.exceptions.NetworkPlanError
        If the pool cannot support a network once the pruned pairs are banned. The
        planner's usual diagnostics apply: it names the components and the rejected
        candidates that would have bridged them.

    Notes
    -----
    **Pruning is expressed as a ban, and the re-plan is the ordinary planner.** That is the
    whole design. Deleting the edges and patching the holes by hand would need a second,
    parallel selection strategy that would drift from the real one; banning them and
    re-running means the replanned network satisfies exactly the same guarantees the first
    one did -- spanning, degree targets, cycle coverage, the CBFE eligibility ladder -- with
    a smaller pool. Nothing new is mapped: the replacements come from the candidates the
    original run already scored and did not select.

    A pruned edge is banned rather than merely dropped because a re-plan over an unmodified
    pool would simply select it again -- it was, after all, the cheapest edge there.

    ``keep_existing`` is expressed the same way, as *forced* edges. Both halves of the
    request therefore travel to the planner as ordinary constraints, and the planner
    resolves them with the machinery it already has -- including refusing outright if a
    surviving edge and a pruned one cannot both be honoured.
    """
    from rbfenetmap.plugins.planners import create_planner

    pruned = select_high_lmi_edges(
        network, lmi, threshold=threshold, quantile=quantile, max_pruned=max_pruned, require_complete=require_complete
    )
    options = _options(network)
    if not pruned:
        logger.info("No edge exceeds the LMI cut; the network is returned unchanged")
        return network, ()

    if not network.candidates:
        raise NetworkPlanError(
            "This network carries no candidate pool, so there is nothing to replan from. Re-plan the "
            "ligands from scratch with these edges in banned_edges: "
            f"{[f'{a}{EDGE_SEPARATOR}{b}' for a, b in pruned]}."
        )

    bans = tuple(dict.fromkeys((*options.banned_edges, *(f"{a}{EDGE_SEPARATOR}{b}" for a, b in pruned))))
    options = replace(options, banned_edges=bans)
    if keep_existing:
        # Only pairs the pool still offers as feasible: forcing an edge the planner cannot
        # supply is a hard error, and a selected CBFE edge has no candidate behind it.
        supplied = {c.unordered_key for c in network.candidates if c.feasible}
        survivors = [
            edge.unordered_key
            for edge in network.edges
            if edge.unordered_key not in set(pruned) and edge.unordered_key in supplied
        ]
        options = replace(
            options,
            forced_edges=tuple(
                dict.fromkeys((*options.forced_edges, *(f"{a}{EDGE_SEPARATOR}{b}" for a, b in survivors)))
            ),
        )

    replanned = create_planner(planner).plan(network.ligands, network.candidates, options)
    logger.info(
        "Replanned after pruning %d edge(s): %d edges before, %d after",
        len(pruned),
        len(network.edges),
        len(replanned.edges),
    )
    return replanned, pruned


def cycle_closure_errors(
    network: Network, values: Mapping[str, float] | Mapping[tuple[str, str], float]
) -> dict[tuple[str, ...], float]:
    """Sum a per-edge quantity around each independent cycle of the network.

    Parameters
    ----------
    network : Network
    values : Mapping
        Per-edge quantities keyed **directionally**, as ``"source~target"`` or as a
        ``(source, target)`` tuple: the value is the quantity for that direction, typically
        the computed ΔΔG. The reverse direction is filled in as its negative, so only one
        orientation of each edge need be supplied.

    Returns
    -------
    dict[tuple[str, ...], float]
        Keyed by the cycle's ligand names in traversal order; the value is the signed sum
        around it, which should be zero and is not.

    Notes
    -----
    Uses a cycle *basis*, not every cycle in the graph: the sums around a basis determine
    the sums around all the rest, and the number of cycles in a dense network is
    exponential. A cycle with an edge missing from *values* is dropped rather than summed
    over what is present, since a partial sum around a loop is not a closure error.

    This is here to make the claim in the module docstring checkable on your own data --
    prune, replan, re-run, and watch these shrink while the errors against experiment do
    not.
    """
    import networkx as nx

    directed: dict[tuple[str, str], float] = {}
    for key, value in values.items():
        source, target = parse_edge_key(key) if isinstance(key, str) else tuple(key)  # type: ignore[misc]
        directed[(source, target)] = float(value)
        directed.setdefault((target, source), -float(value))

    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(network.ligands)
    for edge in network.edges:
        graph.add_edge(edge.source, edge.target)

    closures: dict[tuple[str, ...], float] = {}
    for cycle in nx.cycle_basis(graph):
        total = 0.0
        complete = True
        for index, node in enumerate(cycle):
            partner = cycle[(index + 1) % len(cycle)]
            if (node, partner) not in directed:
                complete = False
                break
            total += directed[(node, partner)]
        if complete:
            closures[tuple(cycle)] = total
    return closures
