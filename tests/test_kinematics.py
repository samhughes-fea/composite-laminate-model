import numpy as np
import pytest

trapezoid = getattr(np, "trapezoid", None) or np.trapz

from laminate_modelling_package import (
    CLPTModel,
    FSDTModel,
    Laminate,
    LayerwiseModel,
    RefinedZigZagModel,
    integrate_abd,
)


@pytest.fixture
def unsymmetric(cfrp):
    return Laminate.from_angles(cfrp, [0, 30, -60, 90], 0.25e-3)


def test_abd_of_symmetric_laminate_has_no_coupling(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 45, -45, 90, 90, -45, 45, 0], 0.125e-3)
    abd = CLPTModel().abd_matrix(lam)
    assert np.allclose(abd[:3, 3:], 0.0, atol=1e-9 * abd[0, 0])
    assert np.allclose(abd, abd.T)


def test_abd_of_single_orthotropic_ply_matches_closed_form(cfrp):
    h = 2e-3
    lam = Laminate.from_angles(cfrp, [0], h)
    abd = integrate_abd(lam)
    Q = cfrp.elastic.reduced_stiffness()
    assert np.allclose(abd[:3, :3], Q * h)
    assert np.allclose(abd[3:, 3:], Q * h**3 / 12)


def test_cross_ply_coupling_b11(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90], 1e-3)
    B = integrate_abd(lam)[:3, 3:]
    Q = cfrp.elastic.reduced_stiffness()
    # B11 = (Q22 - Q11) h^2 / 8 for a two-ply [0/90] with total thickness h = 2 mm... (z^2 terms)
    assert B[0, 0] == pytest.approx((Q[1, 1] - Q[0, 0]) * (1e-3) ** 2 / 2, rel=1e-12)


def test_fsdt_shear_correction(unsymmetric):
    As = FSDTModel().transverse_shear_matrix(unsymmetric)
    raw = sum(p.Q_bar_s * p.thickness for p in unsymmetric.plies)
    assert np.allclose(As, 5.0 / 6.0 * raw)
    assert np.allclose(FSDTModel(1.0).transverse_shear_matrix(unsymmetric), raw)
    with pytest.raises(ValueError):
        FSDTModel(0.0)


def test_clpt_shear_is_penalty(unsymmetric):
    raw = sum(p.Q_bar_s * p.thickness for p in unsymmetric.plies)
    assert np.allclose(CLPTModel(penalty=500).transverse_shear_matrix(unsymmetric), 500 * raw)


def test_rzt_homogeneous_laminate_has_no_zigzag(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 0, 0], 1e-3)
    rzt = RefinedZigZagModel()
    assert np.allclose(rzt.zigzag_slopes(lam), 0.0, atol=1e-12)
    assert np.allclose(rzt.transverse_shear_matrix(lam), sum(p.Q_bar_s * p.thickness for p in lam.plies))


def test_rzt_zigzag_vanishes_on_faces_and_slope_jumps_follow_compliance(unsymmetric):
    rzt = RefinedZigZagModel()
    phi = rzt.zigzag_interface_values(unsymmetric)
    assert np.allclose(phi[0], 0.0) and np.allclose(phi[-1], 0.0, atol=1e-15)
    betas = rzt.zigzag_slopes(unsymmetric)
    jumps = rzt.slope_discontinuities(unsymmetric)
    G = rzt.effective_shear_modulus(unsymmetric)
    S = [np.linalg.inv(p.Q_bar_s) for p in unsymmetric.plies]
    for k in range(1, unsymmetric.n_plies):
        assert np.allclose(jumps[k - 1], betas[k] - betas[k - 1])
        assert np.allclose(jumps[k - 1], (S[k] - S[k - 1]) @ G)


def test_rzt_shear_stress_is_continuous_and_uniform(unsymmetric):
    rzt = RefinedZigZagModel()
    gamma0 = np.array([1e-4, -2e-4])
    G = rzt.effective_shear_modulus(unsymmetric)
    for z in unsymmetric.z_interfaces[1:-1]:
        below = rzt.transverse_shear_stress(unsymmetric, z, gamma0, "below")
        above = rzt.transverse_shear_stress(unsymmetric, z, gamma0, "above")
        assert np.allclose(below, above)
        assert np.allclose(above, G @ gamma0)


def test_fsdt_shear_stress_jumps_at_interfaces(unsymmetric):
    fsdt = FSDTModel()
    gamma0 = np.array([1e-4, 0.0])
    z = unsymmetric.z_interfaces[2]
    assert not np.allclose(
        fsdt.transverse_shear_stress(unsymmetric, z, gamma0, "below"),
        fsdt.transverse_shear_stress(unsymmetric, z, gamma0, "above"),
    )


def test_clpt_parabolic_shear_integrates_to_resultant(unsymmetric):
    clpt = CLPTModel()
    gamma0 = np.array([1e-9, 3e-9])
    resultant = clpt.transverse_shear_matrix(unsymmetric) @ gamma0
    z = np.linspace(-0.5e-3, 0.5e-3, 2001)
    tau = np.array([clpt.transverse_shear_stress(unsymmetric, zi, gamma0) for zi in z])
    assert np.allclose(trapezoid(tau, z, axis=0), resultant, rtol=1e-5)
    assert np.allclose(tau[0], 0.0) and np.allclose(tau[-1], 0.0)


def test_layerwise_projection_reproduces_clpt_abd(unsymmetric):
    assert np.allclose(LayerwiseModel().abd_matrix(unsymmetric), integrate_abd(unsymmetric), rtol=1e-12, atol=1e-6)


def test_layerwise_condensation_matches_rzt(unsymmetric):
    lw, rzt = LayerwiseModel(), RefinedZigZagModel()
    assert np.allclose(lw.transverse_shear_matrix(unsymmetric), rzt.transverse_shear_matrix(unsymmetric), rtol=1e-10)
    for z in unsymmetric.ply_midplanes():
        assert np.allclose(lw.shear_strain_operator(unsymmetric, z), rzt.shear_strain_operator(unsymmetric, z), atol=1e-10)


def test_layerwise_thickness_matrix_layout(unsymmetric):
    tm = LayerwiseModel().thickness_matrices(unsymmetric)
    n_if = unsymmetric.n_plies + 1
    assert tm.membrane.shape == (3 * n_if, 3 * n_if)
    assert tm.shear.shape == (2 * (n_if + 1), 2 * (n_if + 1))
    for matrix in (tm.membrane, tm.shear):
        assert np.allclose(matrix, matrix.T, rtol=0, atol=1e-12 * np.abs(matrix).max())
    # Rigid in-plane translation of every interface carries no shear energy.
    translation = np.concatenate([np.tile([1.0, 0.0], n_if), [0.0, 0.0]])
    assert np.linalg.norm(tm.shear @ translation) < 1e-12 * np.linalg.norm(tm.shear)


def test_layerwise_shear_is_softer_than_uncorrected_fsdt(unsymmetric):
    raw = sum(p.Q_bar_s * p.thickness for p in unsymmetric.plies)
    eig = np.linalg.eigvalsh(raw - LayerwiseModel().transverse_shear_matrix(unsymmetric))
    assert np.all(eig >= -1e-6 * raw.max())


def test_rzt_generalised_stiffness_shapes(unsymmetric):
    out = RefinedZigZagModel().zigzag_stiffness(unsymmetric)
    assert out["B_phi"].shape == (3, 4) and out["D_phiphi"].shape == (4, 4)
    assert np.allclose(out["D_phiphi"], out["D_phiphi"].T)
    assert np.all(np.linalg.eigvalsh(out["D_phiphi"]) >= -1e-12)
