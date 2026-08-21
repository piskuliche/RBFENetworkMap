"""Simple network planners: star, explicit, and complete.

These select without optimising. Each is the right answer to a question the MST planner
answers differently: a star when every transformation must share a reference ligand, an
explicit list when the topology is already decided elsewhere, and the complete graph when
the point is to measure every edge rather than to economise.
"""

from __future__ import annotations

import math
from typing import ClassVar, Mapping, Sequence

from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.meta.planners import AbstractNetworkPlanner
from rbfenetmap.core.models import Ligand, Network, Transformation, parse_edge_key
from rbfenetmap.core.options import HubSelection, NetworkOptions

__all__ = ("CompletePlanner", "ExplicitPlanner", "StarPlanner")


def _feasible_by_pair(candidates: Sequence[Transformation]) -> dict[tuple[str, str], Transformation]:
    """Cheapest feasible orientation of each unordered pair."""
    best: dict[tuple[str, str], Transformation] = {}
    for candidate in candidates:
        if not candidate.feasible:
            continue
        key = candidate.unordered_key
        if key not in best or candidate.score.total < best[key].score.total:
            best[key] = candidate
    return best


class StarPlanner(AbstractNetworkPlanner):
    """Connect every ligand to a single hub.

    The hub defaults to the most central compound in the series. What "central" means is
    ``hub_selection``: the ligand with the most feasible partners (the default), or the one
    with the lowest summed cost to the partners it has.
    """

    name: ClassVar[str] = "star"

    def plan(
        self, ligands: Mapping[str, Ligand], candidates: Sequence[Transformation], options: NetworkOptions
    ) -> Network:
        """Select the hub's spokes."""
        self.check_cbfe_support(options)
        feasible = _feasible_by_pair(candidates)
        feasible = {p: e for p, e in feasible.items() if p not in options.banned_pairs}
        hub = options.hub or self._pick_hub(ligands, feasible, options.hub_selection)
        if hub not in ligands:
            raise NetworkPlanError(f"Hub {hub!r} is not among the ligands.")

        edges = tuple(edge for pair, edge in sorted(feasible.items()) if hub in pair)
        unmet: list[str] = []
        reached = {n for edge in edges for n in (edge.source, edge.target)}
        missing = sorted(set(ligands) - reached - {hub})
        if missing:
            message = f"no feasible edge from hub {hub!r} to {missing}"
            if options.require_connected:
                raise NetworkPlanError(
                    f"Star network cannot span the ligands: {message}. Use the 'mst' planner, "
                    "loosen the soft-core budget, or pass require_connected=False."
                )
            unmet.append(message)

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

    @staticmethod
    def _pick_hub(
        ligands: Mapping[str, Ligand],
        feasible: Mapping[tuple[str, str], Transformation],
        hub_selection: HubSelection = "most_partners",
    ) -> str:
        """Return the hub, by whichever of the two published rules was asked for.

        OpenEye's own documentation concedes that hub choice "is the dominant factor in the
        performance of a star map", which is why this is a knob rather than a constant.

        Parameters
        ----------
        ligands : Mapping[str, Ligand]
        feasible : Mapping[tuple[str, str], Transformation]
            The post-ban feasible pool, keyed by unordered pair.
        hub_selection : {"most_partners", "min_total_cost"}, optional
            ``"most_partners"`` is the historical default: partner count first, cost only
            as a tie-break. More partners beats a lower mean, because a hub reachable from
            only two ligands is no hub at all however cheap those two edges are -- but the
            consequence is that cost is never compared across ligands of differing
            connectivity, so the cheapest well-connected hub can lose to a slightly
            better-connected expensive one.

            ``"min_total_cost"`` is LOMAP's ``pick_lead`` and HiMap's ``ref_lig_gen``:
            lowest summed cost to the partners a ligand actually has, which is the same
            thing as highest summed similarity. Partner count then breaks *its* ties.

        Returns
        -------
        str

        Notes
        -----
        Under ``"min_total_cost"`` a ligand with no feasible partner sums to zero and would
        otherwise win outright, so it is ranked at infinity instead. A hub with no spokes is
        not a cheap hub; it is not a hub.
        """
        totals: dict[str, float] = {name: 0.0 for name in ligands}
        counts: dict[str, int] = {name: 0 for name in ligands}
        for (a, b), edge in feasible.items():
            for name in (a, b):
                totals[name] += edge.score.total
                counts[name] += 1
        if hub_selection == "min_total_cost":
            return min(ligands, key=lambda n: (totals[n] if counts[n] else math.inf, -counts[n], n))
        return min(ligands, key=lambda n: (-counts[n], totals[n], n))


class ExplicitPlanner(AbstractNetworkPlanner):
    """Select exactly the edges named in ``options.explicit_pairs``."""

    name: ClassVar[str] = "explicit"

    def plan(
        self, ligands: Mapping[str, Ligand], candidates: Sequence[Transformation], options: NetworkOptions
    ) -> Network:
        """Select the named edges, failing loudly on any that is unusable."""
        self.check_cbfe_support(options)
        if not options.explicit_pairs:
            raise NetworkPlanError("The 'explicit' planner requires explicit_pairs.")
        feasible = _feasible_by_pair(candidates)

        edges: list[Transformation] = []
        problems: list[str] = []
        for spec in options.explicit_pairs:
            source, target = parse_edge_key(spec)
            key = tuple(sorted((source, target)))
            edge = feasible.get(key)  # type: ignore[arg-type]
            if edge is None:
                problems.append(spec)
                continue
            edges.append(edge if edge.source == source else edge.reversed())

        if problems:
            raise NetworkPlanError(
                f"Explicitly requested edge(s) {problems} are not feasible. The 'explicit' planner "
                "selects exactly what it is given, so it will not silently omit them."
            )

        network = Network(
            ligands=ligands, edges=tuple(edges), candidates=tuple(candidates), planner=self.name, options=options
        )
        network.validate(require_connected=options.require_connected)
        return network


class CompletePlanner(AbstractNetworkPlanner):
    """Select every feasible candidate.

    Maximum redundancy at maximum cost. Useful for small series and for benchmarking a
    sparser network against the full measurement.
    """

    name: ClassVar[str] = "complete"

    def plan(
        self, ligands: Mapping[str, Ligand], candidates: Sequence[Transformation], options: NetworkOptions
    ) -> Network:
        """Select all feasible edges, honouring bans and any edge cap."""
        self.check_cbfe_support(options)
        feasible = _feasible_by_pair(candidates)
        feasible = {p: e for p, e in feasible.items() if p not in options.banned_pairs}
        ordered = sorted(feasible.items(), key=lambda item: (item[1].score.total, item[0]))

        unmet: list[str] = []
        if options.n_edges is not None and len(ordered) > options.n_edges:
            unmet.append(f"n_edges={options.n_edges} truncated {len(ordered)} feasible edges")
            ordered = ordered[: options.n_edges]

        network = Network(
            ligands=ligands,
            edges=tuple(edge for _, edge in ordered),
            candidates=tuple(candidates),
            planner=self.name,
            options=options,
            unmet_constraints=tuple(unmet),
        )
        network.validate(require_connected=options.require_connected)
        return network
