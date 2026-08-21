"""Optimal experimental design over a perturbation network.

The one fact this whole module rests on: **the Fisher information matrix of a network of
relative free energy measurements is the weighted graph Laplacian.**

An RBFE edge measures :math:`\\Delta G_j - \\Delta G_i` with variance
:math:`\\sigma_{ij}^2`. Stacking those observations and forming :math:`X^T \\Sigma^{-1} X`
gives

.. math::

   F_{ii} = \\sum_{k \\ne i} \\sigma_{ik}^{-2}, \\qquad F_{ij} = -\\sigma_{ij}^{-2}

which is exactly the Laplacian of the graph with edge weights :math:`\\sigma_{ij}^{-2}`.
DiffNet, HiMap, Yang's MLE and cinnabar's network analysis are all this same object, so one
implementation serves selection, allocation, and analysis rather than three.

The covariance of the estimated free energies is :math:`C = F^{-1}`, and the two classical
criteria are

- **A-optimal** -- minimise :math:`\\operatorname{tr} C`, the total variance of the
  estimates;
- **D-optimal** -- minimise :math:`\\ln \\det C`, the volume of the joint confidence
  ellipsoid.

Singularity, and why it is not a problem
----------------------------------------
:math:`F` is singular for an RBFE-only network: no relative measurement can pin the
absolute offset, so the all-ones vector is always in the null space. NetBFE regularises by
restraining the mean,

.. math::

   F^*(\\omega) = F + \\omega m^{-2} \\mathbb{1}\\mathbb{1}^T ,

and takes :math:`\\omega \\to \\infty` through the bordered system. Because
:math:`F \\mathbb{1} = 0`, the two terms commute and

.. math::

   \\left(F + \\tfrac{\\omega}{m} P\\right)^{-1} = F^{+} + \\tfrac{m}{\\omega} P
   \\;\\xrightarrow[\\omega \\to \\infty]{}\\; F^{+},

where :math:`P = \\mathbb{1}\\mathbb{1}^T / m` projects onto the null space. So the limit is
just the Moore-Penrose pseudo-inverse, and -- the part that makes the problem well posed --
**the optimal design does not depend on** :math:`\\omega` **at all**. The regulariser only
fixes the unidentifiable offset; it never trades against the criterion.

D-optimality is spanning trees
------------------------------
The pseudo-determinant of a graph Laplacian is :math:`m` times the number of spanning
trees (Kirchhoff's matrix-tree theorem, weighted). Minimising :math:`\\ln \\det C` therefore
*maximises the weighted spanning-tree count*, which is why D-optimal designs come out
markedly more cyclic than A-optimal ones at the same edge count. That is Pitman's argument
for preferring D-optimality whenever a cycle-closure correction will be applied downstream.

What this buys, and what it does not
------------------------------------
Precision, and only precision. Over five TYK2 iterations NetBFE's :math:`\\operatorname{tr}
C` fell monotonically from 1.08 to 0.78 while the RMSE against experiment *rose* from 0.84
to 0.91. An optimal design makes the numbers it produces more reproducible; it says nothing
about whether the force field they come from is right.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

__all__ = (
    "DESIGN_CRITERIA",
    "a_optimal_criterion",
    "a_optimal_gradient",
    "allocate_effort",
    "criterion_value",
    "covariance",
    "d_optimal_criterion",
    "effective_resistances",
    "fisher_information",
    "summarize",
)

#: The criteria :func:`criterion_value` understands, lowest-is-best in both cases.
DESIGN_CRITERIA: tuple[str, ...] = ("a_optimal", "d_optimal")

#: Eigenvalues below this fraction of the largest are treated as null-space. An absolute
#: tolerance would be wrong: the Laplacian's scale is set by the edge weights, which are
#: ``1 / sigma ** 2`` and so vary over orders of magnitude between a 1 kcal/mol edge and a
#: 0.1 kcal/mol one.
_RELATIVE_EIGENVALUE_TOLERANCE = 1e-10


def fisher_information(nodes: Sequence[str], edges: Sequence[tuple[str, str]], sigmas: Sequence[float]) -> np.ndarray:
    """Return the Fisher information matrix of a network, i.e. its weighted Laplacian.

    Parameters
    ----------
    nodes : Sequence[str]
        Ligand names, in the order the matrix rows and columns take.
    edges : Sequence[tuple[str, str]]
        Endpoint pairs. Repeated pairs accumulate, which is correct: two independent
        measurements of the same transformation add their information.
    sigmas : Sequence[float]
        Predicted standard deviation per edge, in the same order. Must be positive.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(len(nodes), len(nodes))`` array.

    Raises
    ------
    ValueError
        If the lengths disagree, an endpoint is not in *nodes*, or a sigma is
        non-positive. A zero sigma is a claim of infinite information from one edge, which
        would make every criterion below read as zero regardless of the rest of the
        network -- silently the best possible design.
    """
    if len(edges) != len(sigmas):
        raise ValueError(f"Got {len(edges)} edge(s) but {len(sigmas)} sigma(s); they must correspond.")
    index = {name: i for i, name in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for (source, target), sigma in zip(edges, sigmas):
        if source not in index or target not in index:
            raise ValueError(f"Edge {source!r}~{target!r} names a ligand outside the node list.")
        if not sigma > 0 or not math.isfinite(sigma):
            raise ValueError(
                f"Edge {source!r}~{target!r} has sigma={sigma!r}. A predicted standard deviation must be "
                "positive and finite; the Fisher weight is 1 / sigma ** 2."
            )
        a, b = index[source], index[target]
        weight = 1.0 / (sigma * sigma)
        matrix[a, a] += weight
        matrix[b, b] += weight
        matrix[a, b] -= weight
        matrix[b, a] -= weight
    return matrix


def covariance(fisher: np.ndarray) -> np.ndarray:
    """Return :math:`C = F^{+}`, the mean-restrained covariance of the estimates.

    Notes
    -----
    The pseudo-inverse *is* the :math:`\\omega \\to \\infty` limit of the bordered system,
    as the module docstring derives -- there is no approximation here and no regularisation
    parameter to choose.
    """
    return np.linalg.pinv(fisher, hermitian=True)


def _nonnull_eigenvalues(fisher: np.ndarray) -> np.ndarray | None:
    """Return the ``n - 1`` non-null eigenvalues, or ``None`` if the network is disconnected.

    The Laplacian's null space has dimension equal to the number of connected components,
    so a nullity above one is exactly the disconnected case -- and there is no need to run
    a separate connectivity check to discover it.
    """
    eigenvalues = np.linalg.eigvalsh(fisher)
    if eigenvalues.size < 2:
        return None
    tolerance = max(float(eigenvalues[-1]), 0.0) * _RELATIVE_EIGENVALUE_TOLERANCE
    positive = eigenvalues[eigenvalues > tolerance]
    if positive.size != eigenvalues.size - 1:
        return None
    return positive


def a_optimal_criterion(fisher: np.ndarray) -> float:
    """Return :math:`\\operatorname{tr} C`, or ``inf`` for a disconnected network.

    ``inf`` rather than a large number: a ligand no edge reaches genuinely has unbounded
    variance, and any finite stand-in would let a disconnected design win a comparison
    against a connected one that happened to be poor.
    """
    positive = _nonnull_eigenvalues(fisher)
    if positive is None:
        return math.inf
    return float(np.sum(1.0 / positive))


def d_optimal_criterion(fisher: np.ndarray) -> float:
    """Return :math:`\\ln \\det C` on the identifiable subspace, or ``inf`` if disconnected.

    Computed as :math:`-\\sum \\ln \\lambda_i` over the non-null eigenvalues -- the log of
    the pseudo-determinant, negated. Summing logs rather than taking a determinant is not
    a micro-optimisation: for a 40-ligand network the product of the eigenvalues underflows
    long before the log of it would.
    """
    positive = _nonnull_eigenvalues(fisher)
    if positive is None:
        return math.inf
    return float(-np.sum(np.log(positive)))


def criterion_value(fisher: np.ndarray, criterion: str) -> float:
    """Evaluate *criterion* on *fisher*. Lower is better for both.

    Raises
    ------
    ValueError
        If *criterion* is not in :data:`DESIGN_CRITERIA`.
    """
    if criterion == "a_optimal":
        return a_optimal_criterion(fisher)
    if criterion == "d_optimal":
        return d_optimal_criterion(fisher)
    raise ValueError(f"Unknown design criterion {criterion!r}. Known: {list(DESIGN_CRITERIA)}.")


def _quadratic_form(matrix: np.ndarray, nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> np.ndarray:
    """Return :math:`u_e^T M u_e` for every edge, with :math:`u_e = e_i - e_j`."""
    index = {name: i for i, name in enumerate(nodes)}
    values = np.empty(len(edges), dtype=float)
    for position, (source, target) in enumerate(edges):
        a, b = index[source], index[target]
        values[position] = matrix[a, a] + matrix[b, b] - 2.0 * matrix[a, b]
    return np.maximum(values, 0.0)


def effective_resistances(nodes: Sequence[str], edges: Sequence[tuple[str, str]], fisher: np.ndarray) -> np.ndarray:
    """Return each edge's effective resistance under the current design.

    Parameters
    ----------
    nodes : Sequence[str]
    edges : Sequence[tuple[str, str]]
    fisher : numpy.ndarray
        The Fisher matrix the resistances are measured against.

    Returns
    -------
    numpy.ndarray
        ``R_e = C_ii + C_jj - 2 C_ij``, one per edge.

    Notes
    -----
    This is the D-optimal gradient, up to sign:
    :math:`\\partial \\ln\\det C / \\partial w_e = -R_e`. It also has the direct reading
    that gives it its name -- an edge with a large effective resistance sits where the
    network is weakest, carrying information no parallel path supplies -- and it sums to
    :math:`n - 1` over any design, which is Foster's theorem and a cheap check that the
    matrix was built right.

    **Not** the A-optimal gradient, which is the same quadratic form taken against
    :math:`C^2` rather than :math:`C`; see :func:`a_optimal_gradient`. Confusing the two is
    easy and self-consistent enough to survive a smoke test, because both are positive and
    both are largest on the same sort of edge -- but only one of them is homogeneous of the
    right degree, so an allocation built on the wrong one drifts instead of converging.
    """
    return _quadratic_form(covariance(fisher), nodes, edges)


def a_optimal_gradient(nodes: Sequence[str], edges: Sequence[tuple[str, str]], fisher: np.ndarray) -> np.ndarray:
    """Return :math:`-\\partial \\operatorname{tr} C / \\partial w_e` for every edge.

    Returns
    -------
    numpy.ndarray
        :math:`u_e^T C^2 u_e`, one per edge, where :math:`u_e = e_i - e_j`.

    Notes
    -----
    From :math:`\\mathrm{d}C = -C \\,\\mathrm{d}F\\, C` and
    :math:`\\partial F/\\partial w_e = u_e u_e^T`, so
    :math:`\\partial \\operatorname{tr} C/\\partial w_e = -\\operatorname{tr}(C u_e u_e^T C)
    = -u_e^T C^2 u_e`. Positive by construction, so more weight on any edge always lowers
    the total variance -- what the design chooses is where the *next* unit of weight helps
    most.
    """
    inverse = covariance(fisher)
    return _quadratic_form(inverse @ inverse, nodes, edges)


def allocate_effort(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    sigmas: Sequence[float],
    *,
    total: float = 1.0,
    iterations: int = 200,
    tolerance: float = 1e-9,
) -> dict[tuple[str, str], float]:
    """Distribute a fixed sampling budget over the edges, A-optimally.

    Parameters
    ----------
    nodes : Sequence[str]
    edges : Sequence[tuple[str, str]]
    sigmas : Sequence[float]
        Predicted standard deviation of each edge **at unit effort**.
    total : float, optional
        Budget to divide, in whatever unit the caller wants back (nanoseconds, GPU-hours,
        or 1.0 for fractions).
    iterations : int, optional
        Cap on the multiplicative iteration.
    tolerance : float, optional
        Stop once the largest relative change in an allocation falls below this.

    Returns
    -------
    dict[tuple[str, str], float]
        Effort per edge, summing to *total*. Keys are the edges as given.

    Raises
    ------
    ValueError
        If *total* is not positive, or the network is disconnected -- an unreachable
        ligand's variance is unbounded, so no finite budget makes the criterion finite and
        there is nothing to optimise.

    Notes
    -----
    The model is the usual one: variance falls as :math:`1/t`, so an edge given effort
    :math:`t_e` contributes Fisher weight :math:`w_e = t_e / v_e` with
    :math:`v_e = \\sigma_e^2` its unit-effort variance. Minimising
    :math:`\\operatorname{tr} C` over :math:`\\sum t_e = T` is convex, and this uses
    Titterington's multiplicative algorithm -- at each step scale every allocation by the
    square root of its directional derivative
    :math:`g_e = u_e^T C^2 u_e / v_e` (see :func:`a_optimal_gradient`) and renormalise. The
    exponent of one half is the standard choice for A-optimality, and it is what makes the
    update scale-invariant: :math:`g_e` is homogeneous of degree :math:`-2` in the effort,
    so :math:`t_e \\sqrt{g_e}` is of degree zero and the iteration cannot drift with the
    budget.

    The stationary point satisfies :math:`g_e = \\operatorname{tr}(C) / T` for every edge,
    which is worth stating in words: **at the optimum every edge returns the same variance
    reduction per nanosecond.** Any other allocation has an edge worth moving time to.

    Published payoff is roughly a twofold variance reduction at equal total cost. This is
    the *static* first pass -- it predicts variances from the descriptors rather than
    measuring them, so it cannot refit against what the simulations actually produced.
    """
    if not total > 0:
        raise ValueError(f"total must be positive; got {total!r}.")
    count = len(edges)
    if count == 0:
        return {}
    variances = np.asarray([float(s) * float(s) for s in sigmas], dtype=float)
    if not np.all(variances > 0):
        raise ValueError("Every sigma must be positive; an edge with zero predicted error absorbs the whole budget.")

    if not math.isfinite(a_optimal_criterion(fisher_information(nodes, edges, sigmas))):
        raise ValueError(
            "Cannot allocate effort over a disconnected network: at least one ligand is unreachable, so "
            "its variance is unbounded at any budget. Plan a connected network first, or allocate over "
            "each component separately."
        )

    effort = np.full(count, total / count, dtype=float)
    for _ in range(iterations):
        fisher = fisher_information(nodes, edges, np.sqrt(variances / effort))
        gradient = a_optimal_gradient(nodes, edges, fisher) / variances
        updated = effort * np.sqrt(np.maximum(gradient, 0.0))
        scale = float(updated.sum())
        if scale <= 0:  # pragma: no cover - only reachable if every resistance is zero
            break
        updated *= total / scale
        shift = float(np.max(np.abs(updated - effort) / np.maximum(effort, 1e-300)))
        effort = updated
        if shift < tolerance:
            break

    return {edge: float(value) for edge, value in zip(edges, effort)}


def summarize(nodes: Sequence[str], edges: Sequence[tuple[str, str]], sigmas: Sequence[float]) -> Mapping[str, float]:
    """Return both criteria for a design, for reporting.

    Returns
    -------
    Mapping[str, float]
        ``{"a_optimal": tr(C), "d_optimal": ln det(C)}``.
    """
    fisher = fisher_information(nodes, edges, sigmas)
    return {"a_optimal": a_optimal_criterion(fisher), "d_optimal": d_optimal_criterion(fisher)}
