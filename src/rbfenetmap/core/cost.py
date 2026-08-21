"""Turning a planner cost into machine time and money, for reporting only.

The scorer's edge total is a *difficulty* number on an arbitrary scale. It orders edges,
which is all selection needs, and it is meaningless to anyone deciding whether a network
fits in the cluster allocation they have this month. This module supplies the other
translation: how many GPU-hours the planned network will take, and what that costs.

**Nothing here feeds selection.** ``cbfe_base_cost`` is untouched, no planner reads a
:class:`CostModel`, and switching ``--cost-units`` cannot move a single edge. That is a
deliberate boundary rather than an omission: a wall-clock price is a constant per edge kind
and would make CBFE edges uniformly unaffordable, which is precisely the confusion the
"eligibility is a gate, not a price" rule in
:doc:`../concepts/network_selection` exists to prevent. Feeding cost into selection is a
later phase's job, where variance-weighted edge costs give it a principled basis.

The defaults come from measurements rather than estimates: Tsai *et al.*, *JCIM* 2026, 66,
1626-1636 (`10.1021/acs.jcim.5c02204
<https://pubs.acs.org/doi/10.1021/acs.jcim.5c02204>`_) Table 1 reports 3.97 GPU-hours for
an RBFE edge at 12 lambda windows and 12.81 for a counterpoised one at 25, a ratio of 3.2.
The price per GPU-hour follows Pitman *et al.*, *JCIM* 2023, 63, 1776-1793, at $0.40.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rbfenetmap.core.models import EdgeKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rbfenetmap.core.models import Network, Transformation

__all__ = ("COST_UNITS", "CostModel", "CostUnits", "network_cost_summary")

CostUnits = Literal["score", "gpu_hours"]

#: The units a cost report can be expressed in. ``"score"`` is the scorer's own arbitrary
#: difficulty scale, which orders edges; ``"gpu_hours"`` is machine time, which orders
#: budgets. Neither is a conversion of the other -- they answer different questions.
COST_UNITS: tuple[CostUnits, ...] = ("score", "gpu_hours")


@dataclass(frozen=True)
class CostModel:
    """Per-edge machine cost, in GPU-hours and in currency.

    Parameters
    ----------
    rbfe_gpu_hours : float
        GPU-hours for one relative edge. The default, 3.97, is Tsai 2026's measurement at
        12 lambda windows.
    cbfe_multiplier : float
        What a counterpoised edge costs relative to a relative one. The default, 3.2, is
        the same paper's 12.81 / 3.97 at 25 lambda windows against 12.

        Expressed as a multiplier rather than as a second absolute figure on purpose: a
        user who changes their lambda schedule or their hardware moves *both* numbers
        together, and the ratio is the part that survives. Overriding
        ``rbfe_gpu_hours`` alone then keeps the CBFE figure honest.
    price_per_gpu_hour : float
        Currency per GPU-hour. The default, 0.40, is the rate Pitman 2023 costs its
        networks at.

    Raises
    ------
    ValueError
        If any field is negative, or ``cbfe_multiplier`` is below one. A counterpoised
        edge is two absolute calculations; it cannot be cheaper than the relative edge it
        replaces, and a multiplier below one would silently invert every report.
    """

    rbfe_gpu_hours: float = 3.97
    cbfe_multiplier: float = 3.2
    price_per_gpu_hour: float = 0.40

    def __post_init__(self) -> None:
        """Reject a model that would produce a nonsensical report."""
        if self.rbfe_gpu_hours < 0:
            raise ValueError("rbfe_gpu_hours must not be negative.")
        if self.price_per_gpu_hour < 0:
            raise ValueError("price_per_gpu_hour must not be negative.")
        if self.cbfe_multiplier < 1.0:
            raise ValueError(
                f"cbfe_multiplier must be at least 1.0, got {self.cbfe_multiplier}. A counterpoised edge "
                "runs two absolute calculations, so it cannot cost less than the relative edge it replaces."
            )

    @property
    def cbfe_gpu_hours(self) -> float:
        """GPU-hours for one counterpoised edge."""
        return self.rbfe_gpu_hours * self.cbfe_multiplier

    def edge_gpu_hours(self, edge: "Transformation") -> float:
        """GPU-hours for one edge, chosen by its :class:`~rbfenetmap.core.models.EdgeKind`."""
        return self.cbfe_gpu_hours if edge.kind is EdgeKind.CBFE else self.rbfe_gpu_hours

    def network_gpu_hours(self, network: "Network") -> float:
        """GPU-hours for every selected edge of *network*."""
        return sum(self.edge_gpu_hours(edge) for edge in network.edges)

    def network_price(self, network: "Network") -> float:
        """Currency cost of running *network*, at :attr:`price_per_gpu_hour`."""
        return self.network_gpu_hours(network) * self.price_per_gpu_hour


def network_cost_summary(network: "Network", *, model: CostModel | None = None) -> dict[str, float]:
    """Summarize what *network* costs, in both the scorer's units and machine time.

    Parameters
    ----------
    network : Network
    model : CostModel, optional
        Defaults to the published figures.

    Returns
    -------
    dict[str, float]
        ``score`` (summed edge totals), ``gpu_hours``, and ``price``. All three are
        reported together so a caller choosing between ``--cost-units`` never has to run
        this twice, and so a reader comparing two networks sees the difficulty figure and
        the wall-clock figure move independently -- which they do, because difficulty
        varies per edge and machine time does not.
    """
    model = model or CostModel()
    return {
        "score": float(sum(edge.score.total for edge in network.edges)),
        "gpu_hours": model.network_gpu_hours(network),
        "price": model.network_price(network),
    }
