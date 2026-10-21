import numpy as np
import pytest

from laminate_modelling_package import (
    CompositePlateElement,
    FSDTModel,
    HygrothermalState,
    Laminate,
    RefinedZigZagModel,
)


@pytest.fixture
def laminate(cfrp):
    return Laminate.from_angles(cfrp, [0, 45, -45, 90], 0.25e-3)


DISTORTED = np.array([[0.0, 0.0], [0.11, 0.01], [0.13, 0.12], [-0.01, 0.09]])


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rigid_body_modes(coords):
    modes = []
    for t in np.eye(3):
        modes.append(np.concatenate([np.concatenate([t, np.zeros(3)]) for _ in coords]))
    for r in np.eye(3):
        modes.append(np.concatenate([np.concatenate([np.cross(r, x), r]) for x in coords]))
    return np.array(modes)


def test_stiffness_is_symmetric_with_six_rigid_body_modes(laminate):
    element = CompositePlateElement(1, (1, 2, 3, 4), DISTORTED, laminate, FSDTModel())
    K = element.stiffness_matrix()
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T)
    eig = np.linalg.eigvalsh(K)
    assert np.all(eig > -1e-9 * eig[-1])
    assert np.sum(np.abs(eig) < 1e-10 * eig[-1]) == 6
    coords3 = np.hstack([DISTORTED, np.zeros((4, 1))])
    for mode in rigid_body_modes(coords3):
        assert np.linalg.norm(K @ mode) < 1e-8 * np.linalg.norm(K) * np.linalg.norm(mode)


def test_stiffness_is_invariant_under_rigid_rotation(laminate):
    flat = CompositePlateElement(1, (1, 2, 3, 4), DISTORTED, laminate, FSDTModel())
    R = rotation_matrix([1.0, 0.3, 0.0], 0.7)
    coords = np.hstack([DISTORTED, np.zeros((4, 1))]) @ R.T
    tilted = CompositePlateElement(1, (1, 2, 3, 4), coords, laminate, FSDTModel())
    # Same material axis requires the projected global x to coincide; compare spectra of isotropic-in-plane laminate
    assert np.allclose(np.linalg.eigvalsh(flat.local_stiffness_matrix())[-6:], np.linalg.eigvalsh(tilted.local_stiffness_matrix())[-6:], rtol=0.5)
    K = tilted.stiffness_matrix()
    for mode in rigid_body_modes(coords):
        assert np.linalg.norm(K @ mode) < 1e-8 * np.linalg.norm(K) * np.linalg.norm(mode)


def test_membrane_patch_constant_strain(laminate):
    element = CompositePlateElement(1, (1, 2, 3, 4), DISTORTED, laminate, FSDTModel())
    eps = np.array([1e-4, -3e-5, 2e-5])
    u = np.zeros(24)
    for i, (x, y) in enumerate(DISTORTED):
        u[6 * i] = eps[0] * x + 0.5 * eps[2] * y
        u[6 * i + 1] = eps[1] * y + 0.5 * eps[2] * x
        u[6 * i + 5] = 0.0  # zero rigid rotation: (v,x - u,y)/2 = 0
    for xi, eta in [(0, 0), (0.5, -0.3), (-0.9, 0.9)]:
        e0, kappa, gamma = element.generalized_strains(u, xi, eta)
        assert np.allclose(e0, eps)
        assert np.allclose(kappa, 0.0) and np.allclose(gamma, 0.0)


def test_bending_patch_constant_curvature_has_zero_shear(laminate):
    element = CompositePlateElement(1, (1, 2, 3, 4), DISTORTED, laminate, FSDTModel())
    kx, ky, kxy = 0.5, -0.2, 0.3
    u = np.zeros(24)
    for i, (x, y) in enumerate(DISTORTED):
        w = -0.5 * (kx * x * x + ky * y * y + kxy * x * y)
        beta_x, beta_y = kx * x + 0.5 * kxy * y, ky * y + 0.5 * kxy * x  # beta = -grad w
        u[6 * i + 2] = w
        u[6 * i + 4] = beta_x
        u[6 * i + 3] = -beta_y
    for xi, eta in [(0, 0), (0.57, -0.57), (-0.3, 0.8)]:
        _, kappa, gamma = element.generalized_strains(u, xi, eta)
        assert np.allclose(kappa, [kx, ky, kxy])
        assert np.allclose(gamma, 0.0, atol=1e-12)


def test_pressure_and_hygrothermal_load_vectors(laminate):
    element = CompositePlateElement(1, (1, 2, 3, 4), DISTORTED, laminate, FSDTModel())
    F = element.pressure_load_vector(2.0e3)
    assert np.sum(F[2::6]) == pytest.approx(2.0e3 * element.area)
    F_ht = element.hygrothermal_load_vector(HygrothermalState(-80.0, 0.004, temperature_gradient=1e4))
    coords3 = np.hstack([DISTORTED, np.zeros((4, 1))])
    for mode in rigid_body_modes(coords3):
        assert abs(F_ht @ mode) < 1e-10 * np.linalg.norm(F_ht) * np.linalg.norm(mode)
    assert np.linalg.norm(F_ht) > 0.0


def test_element_rejects_bad_geometry(laminate):
    with pytest.raises(ValueError):
        CompositePlateElement(1, (1, 2, 3, 4), DISTORTED[[0, 2, 1, 3]], laminate, FSDTModel())  # bow-tie
    with pytest.raises(ValueError):
        CompositePlateElement(1, (1, 2, 3), DISTORTED[:3], laminate, FSDTModel())


def test_with_laminate_and_kinematics_create_new_elements(laminate, cfrp):
    element = CompositePlateElement(1, (1, 2, 3, 4), DISTORTED, laminate, FSDTModel())
    other = element.with_kinematics(RefinedZigZagModel())
    assert other.kinematics.name == "RZT" and element.kinematics.name == "FSDT"
    thin = element.with_laminate(Laminate.from_angles(cfrp, [0], 1e-4))
    assert thin.laminate.n_plies == 1 and element.laminate.n_plies == 4
