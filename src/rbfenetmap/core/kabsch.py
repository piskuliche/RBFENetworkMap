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

__all__ = ("apply_transform", "core_rmsd", "kabsch_rotation", "pair_distances", "rigid_transform", "superpose")


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


def rigid_transform(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(rotation, translation)`` best superposing *mobile* onto *reference*.

    Parameters
    ----------
    mobile, reference : numpy.ndarray
        ``(n, 3)`` coordinate arrays of corresponding points.

    Returns
    -------
    rotation : numpy.ndarray
        A ``(3, 3)`` proper rotation matrix.
    translation : numpy.ndarray
        A length-3 vector, such that ``mobile @ rotation.T + translation`` is the
        superposed result.

    Raises
    ------
    ValueError
        If the two arrays do not have the same shape.

    Notes
    -----
    This is the piece :func:`superpose` cannot give you, and the reason it exists. The fit
    is computed over a *subset* of corresponding atoms -- typically a common core -- but the
    transform it yields is a property of the whole rigid body, so it can be applied to every
    atom of the mobile molecule, including the ones that took no part in the fit. Superposing
    coordinate arrays in isolation re-centres only the points it was handed, which is correct
    for measuring an RMSD and useless for moving a molecule.
    """
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.shape != reference.shape:
        raise ValueError(f"Coordinate shapes disagree: {mobile.shape} vs {reference.shape}.")
    if mobile.shape[0] == 0:
        return np.eye(3), np.zeros(3)
    mobile_centre = mobile.mean(axis=0)
    reference_centre = reference.mean(axis=0)
    rotation = kabsch_rotation(mobile - mobile_centre, reference - reference_centre)
    # (m - m_bar) @ R.T + r_bar == m @ R.T + (r_bar - m_bar @ R.T), so the centring folds
    # into a single translation and the transform becomes applicable to any point.
    return rotation, reference_centre - mobile_centre @ rotation.T


def apply_transform(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Return ``coords`` rotated and translated by a :func:`rigid_transform` result.

    Parameters
    ----------
    coords : numpy.ndarray
        ``(n, 3)`` coordinates. Need not be the points the transform was fitted on.
    rotation : numpy.ndarray
        A ``(3, 3)`` rotation matrix.
    translation : numpy.ndarray
        A length-3 translation vector.

    Returns
    -------
    numpy.ndarray
        The transformed ``(n, 3)`` coordinates.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 0:
        return coords.copy()
    return coords @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)


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

    See Also
    --------
    rigid_transform : Returns the transform itself, for applying to further points.
    """
    return apply_transform(np.asarray(mobile, dtype=float), *rigid_transform(mobile, reference))


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

    Superposing here measures a single pair in isolation and throws the transform away. To
    bring a whole *set* of ligands into a common frame before planning -- the case where the
    inputs were prepared separately, for instance converted to mol2 from independent Amber
    topologies -- use :mod:`rbfenetmap.core.align` instead.
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
