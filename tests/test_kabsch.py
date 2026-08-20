"""Tests for the rigid-body superposition helpers.

These functions carry a load-bearing claim in their docstrings that nothing verified until
now: ``kabsch_rotation`` returns a *proper* rotation, so a degenerate core cannot superpose
onto its own mirror image and win a flatteringly low RMSD. The chirality test below is that
claim, made executable.

The rest of the file pins the algebraic identity that
:mod:`rbfenetmap.core.align` depends on -- that a transform fitted on a subset of atoms can
be applied to atoms that took no part in the fit, and moves them rigidly.
"""

from __future__ import annotations

import numpy as np
import pytest

from rbfenetmap.core.kabsch import (
    apply_transform,
    core_rmsd,
    kabsch_rotation,
    pair_distances,
    rigid_transform,
    superpose,
)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """A proper rotation of *angle* radians about *axis*, by Rodrigues' formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _cloud(n: int = 12, seed: int = 7) -> np.ndarray:
    """A deterministic, non-degenerate point cloud."""
    return np.random.default_rng(seed).normal(size=(n, 3)) * 3.0


class TestKabschRotation:
    def test_recovers_a_known_rotation(self):
        reference = _cloud()
        reference -= reference.mean(axis=0)
        rotation = _rotation([0.3, -0.7, 0.6], 1.1)
        mobile = reference @ rotation
        recovered = kabsch_rotation(mobile, reference)
        assert np.allclose(mobile @ recovered.T, reference, atol=1e-9)

    def test_result_is_a_proper_rotation(self):
        reference = _cloud()
        reference -= reference.mean(axis=0)
        mobile = _cloud(seed=11)
        mobile -= mobile.mean(axis=0)
        rotation = kabsch_rotation(mobile, reference)
        assert np.isclose(np.linalg.det(rotation), 1.0)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)

    def test_a_chiral_set_is_not_superposed_onto_its_mirror_image(self):
        # A genuinely three-dimensional cloud and its reflection are related only by a
        # rotoinversion. Without the determinant correction the SVD would happily return
        # one, and a chirally wrong mapping would score as a perfect fit.
        cloud = _cloud()
        cloud -= cloud.mean(axis=0)
        mirrored = cloud * np.array([1.0, 1.0, -1.0])
        assert core_rmsd(mirrored, cloud, superpose_first=True) > 1.0

    def test_a_planar_set_superposes_onto_its_in_plane_reflection(self):
        # The flip side, and the reason the module's Notes single out planar cores: an
        # in-plane reflection of a flat set *is* a proper rotation, so no determinant
        # correction can distinguish it. Recorded here so the limitation is documented
        # rather than discovered.
        planar = _cloud()
        planar[:, 2] = 0.0
        planar -= planar.mean(axis=0)
        mirrored = planar * np.array([1.0, -1.0, 1.0])
        assert core_rmsd(mirrored, planar, superpose_first=True) < 1e-9

    def test_a_degenerate_covariance_still_returns_a_rotation(self):
        collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        collinear -= collinear.mean(axis=0)
        rotation = kabsch_rotation(collinear, collinear)
        assert np.isclose(abs(np.linalg.det(rotation)), 1.0)


class TestRigidTransform:
    def test_agrees_with_superpose(self):
        mobile, reference = _cloud(seed=1), _cloud(seed=2)
        assert np.allclose(superpose(mobile, reference), apply_transform(mobile, *rigid_transform(mobile, reference)))

    def test_recovers_a_known_frame_change(self):
        reference = _cloud()
        rotation = _rotation([1.0, 2.0, -0.5], 2.4)
        offset = np.array([11.0, -4.0, 130.0])
        mobile = reference @ rotation.T + offset
        recovered_rotation, recovered_translation = rigid_transform(mobile, reference)
        moved = apply_transform(mobile, recovered_rotation, recovered_translation)
        assert np.allclose(moved, reference, atol=1e-9)

    def test_moves_points_outside_the_fitted_subset_rigidly(self):
        # The whole feature rests on this: fit on a common core, move every atom.
        whole = _cloud(n=20, seed=3)
        rotation = _rotation([0.2, 0.9, 0.3], 0.8)
        offset = np.array([-7.0, 2.0, 40.0])
        displaced = whole @ rotation.T + offset

        fit_subset = slice(0, 8)
        transform = rigid_transform(displaced[fit_subset], whole[fit_subset])
        moved = apply_transform(displaced, *transform)

        assert np.allclose(moved, whole, atol=1e-9)
        before = pair_distances(displaced[:-1], displaced[1:])
        after = pair_distances(moved[:-1], moved[1:])
        assert np.allclose(before, after)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes disagree"):
            rigid_transform(np.zeros((3, 3)), np.zeros((4, 3)))

    def test_empty_input_is_the_identity(self):
        rotation, translation = rigid_transform(np.zeros((0, 3)), np.zeros((0, 3)))
        assert np.allclose(rotation, np.eye(3))
        assert np.allclose(translation, np.zeros(3))
        assert apply_transform(np.zeros((0, 3)), rotation, translation).shape == (0, 3)


class TestCoreRmsd:
    def test_in_place_rmsd_is_large_for_a_displaced_frame(self):
        # The one-line summary of why pre-alignment exists.
        reference = _cloud()
        displaced = reference @ _rotation([0.0, 0.0, 1.0], 1.3).T + np.array([25.0, 0.0, 0.0])
        assert core_rmsd(displaced, reference) > 10.0
        assert core_rmsd(displaced, reference, superpose_first=True) < 1e-9

    def test_empty_core_is_zero(self):
        assert core_rmsd(np.zeros((0, 3)), np.zeros((0, 3))) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes disagree"):
            core_rmsd(np.zeros((2, 3)), np.zeros((3, 3)))


class TestPairDistances:
    def test_known_values(self):
        mobile = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        reference = np.zeros((2, 3))
        assert np.allclose(pair_distances(mobile, reference), [0.0, 5.0])

    def test_empty_input(self):
        assert pair_distances(np.zeros((0, 3)), np.zeros((0, 3))).shape == (0,)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes disagree"):
            pair_distances(np.zeros((2, 3)), np.zeros((3, 3)))
