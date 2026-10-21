import numpy as np
import pytest

from laminate_modelling_package import (
    Laminate,
    Ply,
    q_bar_from_invariants,
    rotate_stiffness_3d,
    transform_reduced_stiffness,
)


@pytest.mark.parametrize("theta", np.linspace(-90, 90, 13))
def test_q_bar_reuter_equals_invariant_form(cfrp, theta):
    ply = Ply(cfrp, 1e-3, theta)
    assert np.allclose(ply.Q_bar, q_bar_from_invariants(ply.Q, theta), rtol=1e-12, atol=1e-3)


def test_q_bar_special_angles(cfrp):
    Q = cfrp.elastic.reduced_stiffness()
    assert np.allclose(Ply(cfrp, 1e-3, 0).Q_bar, Q)
    Q90 = Ply(cfrp, 1e-3, 90).Q_bar
    assert Q90[0, 0] == pytest.approx(Q[1, 1])
    assert Q90[1, 1] == pytest.approx(Q[0, 0])
    assert abs(Q90[0, 2]) < 1e-6 * Q[0, 0]


def test_q_bar_matches_rotated_3d_stiffness(cfrp):
    theta = 30.0
    C_bar = rotate_stiffness_3d(cfrp.elastic.stiffness_matrix(), theta)
    S_bar = np.linalg.inv(C_bar)
    idx = [0, 1, 5]
    assert np.allclose(np.linalg.inv(S_bar[np.ix_(idx, idx)]), transform_reduced_stiffness(cfrp.elastic.reduced_stiffness(), theta))


def test_transverse_shear_rotation(cfrp):
    e = cfrp.elastic
    theta = 35.0
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    Qs = Ply(cfrp, 1e-3, theta).Q_bar_s
    assert Qs[0, 0] == pytest.approx(e.G23 * c * c + e.G13 * s * s)
    assert Qs[1, 1] == pytest.approx(e.G13 * c * c + e.G23 * s * s)
    assert Qs[0, 1] == pytest.approx((e.G13 - e.G23) * c * s)


def test_ply_matrices_are_read_only(cfrp):
    ply = Ply(cfrp, 1e-3, 45)
    with pytest.raises(ValueError):
        ply.Q_bar[0, 0] = 0.0
    with pytest.raises(ValueError):
        Ply(cfrp, -1e-3, 0)


def test_stress_round_trip(cfrp):
    ply = Ply(cfrp, 1e-3, 40)
    sigma = np.array([10.0, -3.0, 4.0])
    assert np.allclose(np.linalg.solve(ply.T, ply.to_material_stress(sigma)), sigma)


def test_laminate_geometry(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 45, -45, 90], 0.25e-3)
    assert lam.thickness == pytest.approx(1e-3)
    assert np.allclose(lam.z_interfaces, [-0.5e-3, -0.25e-3, 0, 0.25e-3, 0.5e-3])
    assert lam.ply_index_at(-0.5e-3) == 0
    assert lam.ply_index_at(0.0, "above") == 2
    assert lam.ply_index_at(0.0, "below") == 1
    assert lam.ply_index_at(0.5e-3) == 3
    with pytest.raises(ValueError):
        lam.ply_index_at(0.6e-3)
    assert lam.stacking_sequence == "[0/45/-45/90]"
    assert not lam.is_symmetric()
    assert lam.is_balanced()
    assert Laminate.from_angles(cfrp, [0, 90, 90, 0], 1e-4).is_symmetric()
    assert not Laminate.from_angles(cfrp, [0, 45], 1e-4).is_balanced()


def test_replace_ply_returns_new_laminate(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90], 1e-4)
    new = lam.replace_ply(1, Ply(cfrp, 1e-4, 45))
    assert lam.plies[1].angle == 90 and new.plies[1].angle == 45
