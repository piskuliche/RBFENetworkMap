"""Rigid-body superposition and core-geometry metrics.

NumPy only -- no RDKit, no ParmEd. Callers pass coordinate arrays extracted from
whatever molecule representation they hold.

The core RMSD computed here is a genuine quality signal for a mapping, not just a
diagnostic. Two molecules can share a large maximum common substructure whose atoms sit
in completely different places once the ligands are posed in the binding site; such a
mapping is topologically defensible and physically useless. A high core RMSD is how that
shows up.
"""

from __future__ import annotations

import numpy as np

__all__ = ("core_rmsd", "kabsch_rotation", "pair_distances", "superpose")


def kabsch_rotation(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the rotation matrix best superposing *mobile* onto *reference*.

    Both arrays must already be centred on their own centroids.

    Parameters
    ----------
    mobile, reference : numpy.ndarray
        ``(n, 3)`` centred coordinate arrays.

    Returns
    -------
    numpy.ndarray
        A ``(3, 3)`` proper rotation matrix (determinant ``+1``).

    Notes
    -----
    The determinant correction is what keeps this a rotation rather than a rotoinversion.
    Without it, a near-planar or otherwise degenerate core can superpose onto its own
    mirror image, giving a flatteringly low RMSD for a mapping that is in fact chirally
    wrong.
    """
    covariance = mobile.T @ reference
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign if sign != 0 else 1.0])
    return vt.T @ correction @ u.T


def superpose(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return *mobile* rigidly superposed onto *reference*.

    Parameters
    ----------
    mobile, reference : numpy.ndarray
        ``(n, 3)`` coordinate arrays of corresponding points.

    Returns
    -------
    numpy.ndarray
        The transformed *mobile* coordinates.
    """
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.shape != reference.shape:
        raise ValueError(f"Coordinate shapes disagree: {mobile.shape} vs {reference.shape}.")
    if mobile.shape[0] == 0:
        return mobile.copy()
    mobile_centre = mobile.mean(axis=0)
    reference_centre = reference.mean(axis=0)
    rotation = kabsch_rotation(mobile - mobile_centre, reference - reference_centre)
    return (mobile - mobile_centre) @ rotation.T + reference_centre


def core_rmsd(mobile: np.ndarray, reference: np.ndarray, *, superpose_first: bool = False) -> float:
    """RMSD between corresponding points.

    Parameters
    ----------
    mobile, reference : numpy.ndarray
        ``(n, 3)`` coordinate arrays of corresponding atoms.
    superpose_first : bool, optional
        Whether to rigidly superpose before measuring. Default ``False``.

    Returns
    -------
    float
        The RMSD, or ``0.0`` for an empty core.

    Notes
    -----
    The default is *not* to superpose. Ligands are normally supplied already posed in a
    common binding-site frame, and in that frame the in-place deviation of the mapped
    core is the physically meaningful quantity: it says whether the mapping pairs atoms
    that actually occupy the same region of the pocket. Superposing first would discard
    precisely that information and reward a mapping that is self-consistent but
    misplaced. Pass ``superpose_first=True`` only when the inputs are not co-posed.
    """
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.shape != reference.shape:
        raise ValueError(f"Coordinate shapes disagree: {mobile.shape} vs {reference.shape}.")
    if mobile.shape[0] == 0:
        return 0.0
    if superpose_first:
        mobile = superpose(mobile, reference)
    return float(np.sqrt(np.mean(np.sum((mobile - reference) ** 2, axis=1))))


def pair_distances(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-pair distances between corresponding points.

    Returns
    -------
    numpy.ndarray
        A length-``n`` array. Useful for finding the single worst-placed mapped pair,
        which is often more diagnostic than the aggregate RMSD.
    """
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.shape != reference.shape:
        raise ValueError(f"Coordinate shapes disagree: {mobile.shape} vs {reference.shape}.")
    if mobile.shape[0] == 0:
        return np.zeros(0, dtype=float)
    return np.sqrt(np.sum((mobile - reference) ** 2, axis=1))
