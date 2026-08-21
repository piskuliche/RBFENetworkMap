"""The Fisher-information layer: :mod:`rbfenetmap.core.design`.

Every case here is checkable by hand or against a closed form. A triangle of unit-sigma
edges has Laplacian eigenvalues ``{0, 3, 3}``, so ``tr(C) = 2/3`` and
``ln det(C) = -2 ln 3`` -- no tolerance-chasing and nothing that a change in numpy would
quietly shift.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rbfenetmap.core.design import (
    a_optimal_criterion,
    a_optimal_gradient,
    allocate_effort,
    covariance,
    criterion_value,
    d_optimal_criterion,
    effective_resistances,
    fisher_information,
    summarize,
)

TRIANGLE = ["a", "b", "c"]
TRIANGLE_EDGES = [("a", "b"), ("b", "c"), ("a", "c")]


class TestFisherInformation:
    def test_is_the_weighted_laplacian(self):
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.0, 1.0])
        assert np.allclose(fisher, np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float))

    def test_weights_are_the_inverse_variance(self):
        fisher = fisher_information(["a", "b"], [("a", "b")], [2.0])
        assert fisher[0, 0] == pytest.approx(0.25)

    def test_rows_sum_to_zero(self):
        """The defining property of a Laplacian, and the reason the ones vector is null."""
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, [0.5, 1.5, 2.5])
        assert np.allclose(fisher.sum(axis=1), 0.0)

    def test_repeated_pairs_accumulate(self):
        """Two independent measurements of one transformation add their information."""
        once = fisher_information(["a", "b"], [("a", "b")], [1.0])
        twice = fisher_information(["a", "b"], [("a", "b"), ("a", "b")], [1.0, 1.0])
        assert np.allclose(twice, 2 * once)

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="must correspond"):
            fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0])

    def test_an_unknown_endpoint_is_refused(self):
        with pytest.raises(ValueError, match="outside the node list"):
            fisher_information(["a", "b"], [("a", "z")], [1.0])

    @pytest.mark.parametrize("sigma", [0.0, -1.0, math.inf, math.nan])
    def test_a_non_positive_sigma_is_refused(self, sigma):
        """A zero sigma claims infinite information and would win every comparison."""
        with pytest.raises(ValueError, match="positive and finite"):
            fisher_information(["a", "b"], [("a", "b")], [sigma])


class TestCriteria:
    def test_a_optimal_matches_the_closed_form(self):
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.0, 1.0])
        assert a_optimal_criterion(fisher) == pytest.approx(2.0 / 3.0)

    def test_d_optimal_matches_the_closed_form(self):
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.0, 1.0])
        assert d_optimal_criterion(fisher) == pytest.approx(-2.0 * math.log(3.0))

    def test_the_pseudo_determinant_counts_spanning_trees(self):
        """Kirchhoff: pdet(L) = n x (weighted spanning trees). A triangle has three."""
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.0, 1.0])
        assert math.exp(-d_optimal_criterion(fisher)) == pytest.approx(len(TRIANGLE) * 3)

    def test_adding_an_edge_never_worsens_either_criterion(self):
        path = fisher_information(TRIANGLE, TRIANGLE_EDGES[:2], [1.0, 1.0])
        triangle = fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.0, 1.0])
        assert a_optimal_criterion(triangle) < a_optimal_criterion(path)
        assert d_optimal_criterion(triangle) < d_optimal_criterion(path)

    @pytest.mark.parametrize("criterion", ["a_optimal", "d_optimal"])
    def test_a_disconnected_network_is_infinite(self, criterion):
        """Not merely large. A ligand no edge reaches genuinely has unbounded variance."""
        fisher = fisher_information(TRIANGLE, [("a", "b")], [1.0])
        assert criterion_value(fisher, criterion) == math.inf

    def test_an_unknown_criterion_is_refused(self):
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="Unknown design criterion"):
            criterion_value(fisher, "e_optimal")

    def test_summarize_reports_both(self):
        both = summarize(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.0, 1.0])
        assert set(both) == {"a_optimal", "d_optimal"}
        assert both["a_optimal"] == pytest.approx(2.0 / 3.0)


class TestRegularisation:
    def test_the_pseudo_inverse_is_the_omega_limit(self):
        """The bordered system converges on ``pinv``, so no regulariser has to be chosen."""
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.3, 0.7])
        ones = np.ones((3, 3))
        for omega in (1e6, 1e9, 1e12):
            bordered = np.linalg.inv(fisher + omega * ones / 9.0)
            assert np.allclose(bordered, covariance(fisher), atol=10.0 / omega)

    def test_the_covariance_is_mean_centred(self):
        """``C 1 = 0``: the absolute offset is unidentifiable, and the limit says so."""
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.3, 0.7])
        assert np.allclose(covariance(fisher) @ np.ones(3), 0.0, atol=1e-10)


class TestGradients:
    def test_effective_resistances_obey_fosters_theorem(self):
        """``sum w_e R_e = n - 1`` over any connected design -- a free check on the matrix."""
        sigmas = [1.0, 1.3, 0.7]
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, sigmas)
        weights = np.array([1.0 / (s * s) for s in sigmas])
        resistances = effective_resistances(TRIANGLE, TRIANGLE_EDGES, fisher)
        assert float(weights @ resistances) == pytest.approx(len(TRIANGLE) - 1)

    def test_the_a_optimal_gradient_matches_a_finite_difference(self):
        """``d tr(C) / d w_e = -u^T C^2 u``, checked against perturbing one weight."""
        sigmas = [1.0, 1.3, 0.7]
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, sigmas)
        analytic = a_optimal_gradient(TRIANGLE, TRIANGLE_EDGES, fisher)
        step = 1e-6
        for position in range(len(TRIANGLE_EDGES)):
            weights = [1.0 / (s * s) for s in sigmas]
            weights[position] += step
            bumped = fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0 / math.sqrt(w) for w in weights])
            numeric = (a_optimal_criterion(bumped) - a_optimal_criterion(fisher)) / step
            assert numeric == pytest.approx(-analytic[position], rel=1e-4)

    def test_the_two_gradients_are_different(self):
        """C against C^2 -- easy to confuse, and only one has the right homogeneity."""
        fisher = fisher_information(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.3, 0.7])
        assert not np.allclose(
            effective_resistances(TRIANGLE, TRIANGLE_EDGES, fisher),
            a_optimal_gradient(TRIANGLE, TRIANGLE_EDGES, fisher),
        )


class TestAllocation:
    def test_a_symmetric_network_is_allocated_uniformly(self):
        effort = allocate_effort(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.0, 1.0], total=3.0)
        assert all(value == pytest.approx(1.0) for value in effort.values())

    def test_a_path_allocates_in_proportion_to_variance(self):
        """Two edges in series: the closed form is ``t_e proportional to v_e``."""
        effort = allocate_effort(TRIANGLE, [("a", "b"), ("b", "c")], [1.0, 2.0], total=3.0)
        assert effort[("a", "b")] == pytest.approx(1.0, rel=1e-6)
        assert effort[("b", "c")] == pytest.approx(2.0, rel=1e-6)

    def test_the_budget_is_spent_exactly(self):
        effort = allocate_effort(TRIANGLE, TRIANGLE_EDGES, [0.4, 1.9, 1.1], total=250.0)
        assert sum(effort.values()) == pytest.approx(250.0)

    def test_it_beats_a_uniform_split(self):
        """The point of the exercise: less total variance for the same machine time."""
        nodes = [f"L{i}" for i in range(6)]
        edges = [("L0", "L1"), ("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("L4", "L5"), ("L5", "L0"), ("L0", "L3")]
        sigmas = [0.4, 1.8, 0.6, 2.2, 0.5, 1.1, 1.6]
        total = float(len(edges))

        def trace(effort):
            scaled = [s / math.sqrt(t) for s, t in zip(sigmas, effort)]
            return a_optimal_criterion(fisher_information(nodes, edges, scaled))

        optimal = allocate_effort(nodes, edges, sigmas, total=total)
        assert trace([optimal[e] for e in edges]) < trace([1.0] * len(edges))

    def test_at_the_optimum_every_funded_edge_returns_the_same_reduction(self):
        """The KKT condition, stated as the thing it means.

        Every edge with time on it returns the same variance reduction per nanosecond; an
        edge returning *less* than that gets no time at all. The second half is not a
        degenerate case to be tolerated -- it is the allocation telling the user that an
        edge the planner selected is redundant given how the rest was funded.
        """
        nodes = ["a", "b", "c", "d"]
        edges = [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "c")]
        sigmas = [0.5, 1.4, 0.9, 2.0, 1.1]
        total = 10.0
        effort = allocate_effort(nodes, edges, sigmas, total=total)
        scaled = [s / math.sqrt(effort[e]) for s, e in zip(sigmas, edges)]
        fisher = fisher_information(nodes, edges, scaled)
        variances = np.array([s * s for s in sigmas])
        gradient = a_optimal_gradient(nodes, edges, fisher) / variances
        multiplier = a_optimal_criterion(fisher) / total

        assert np.all(gradient <= multiplier * (1.0 + 1e-3))
        funded = np.array([effort[e] > 0.01 * total for e in edges])
        assert funded.any()
        assert np.allclose(gradient[funded], multiplier, rtol=1e-3)

    def test_an_empty_design_allocates_nothing(self):
        assert allocate_effort(TRIANGLE, [], [], total=1.0) == {}

    def test_a_non_positive_budget_is_refused(self):
        with pytest.raises(ValueError, match="total must be positive"):
            allocate_effort(TRIANGLE, TRIANGLE_EDGES, [1.0, 1.0, 1.0], total=0.0)

    def test_a_disconnected_network_is_refused(self):
        with pytest.raises(ValueError, match="disconnected network"):
            allocate_effort(TRIANGLE, [("a", "b")], [1.0], total=1.0)
