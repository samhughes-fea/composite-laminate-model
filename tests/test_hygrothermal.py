import numpy as np
import pytest

trapezoid = getattr(np, "trapezoid", None) or np.trapz

from composite_laminate_model import (
    DOF,
    BoundaryConditions,
    ExactStressRecoverer,
    FEAMesh,
    FSDTModel,
    HygrothermalEngine,
    HygrothermalProperties,
    HygrothermalState,
    Laminate,
    LoadCase,
    Ply,
    StructuralGlobalSolver,
    integrate_abd,
)


def test_global_expansion_coefficients(cfrp):
    engine = HygrothermalEngine()
    theta = 30.0
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    alpha, beta = engine.global_expansion_coefficients(Ply(cfrp, 1e-3, theta))
    a1, a2 = cfrp.hygrothermal.alpha1, cfrp.hygrothermal.alpha2
    assert np.allclose(alpha, [a1 * c * c + a2 * s * s, a1 * s * s + a2 * c * c, 2 * (a1 - a2) * c * s])
    b2 = cfrp.hygrothermal.beta2
    assert np.allclose(beta, [b2 * s * s, b2 * c * c, -2 * b2 * c * s])


def test_properties_defaults():
    p = HygrothermalProperties(1e-6, 2e-5, beta2=0.3)
    assert p.alpha3 == 2e-5 and p.beta3 == 0.3
    assert np.allclose(p.alpha_vector(), [1e-6, 2e-5, 2e-5, 0, 0, 0])


def test_resultants_match_numerical_integration(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 45, 90, -30], 0.3e-3)
    state = HygrothermalState(-120.0, 0.01, temperature_gradient=2e4, moisture_gradient=-3.0)
    engine = HygrothermalEngine()
    N, M = engine.resultants(lam, state)
    N_num, M_num = np.zeros(3), np.zeros(3)
    for ply, (zb, zt) in zip(lam.plies, lam.ply_bounds()):
        z = np.linspace(zb, zt, 401)
        s = np.array([ply.Q_bar @ engine.global_free_strain(ply, state, zi) for zi in z])
        N_num += trapezoid(s, z, axis=0)
        M_num += trapezoid(s * z[:, None], z, axis=0)
    assert np.allclose(N, N_num, rtol=1e-6)
    assert np.allclose(M, M_num, rtol=1e-5)


def test_symmetric_laminate_uniform_state_has_no_thermal_moment(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.25e-3)
    _, M = HygrothermalEngine().resultants(lam, HygrothermalState(-100.0))
    assert np.allclose(M, 0.0, atol=1e-9)


def free_plate(laminate, n=4, a=0.1):
    mesh = FEAMesh.rectangular_plate(a, a, n, n, laminate, FSDTModel())
    c0, c1, c2 = mesh.closest_node((0, 0)), mesh.closest_node((a, 0)), mesh.closest_node((0, a))
    bcs = BoundaryConditions().fix([c0], [DOF.UX, DOF.UY, DOF.UZ]).fix([c1], [DOF.UY, DOF.UZ]).fix([c2], [DOF.UZ])
    return mesh, bcs


@pytest.mark.parametrize("stack", [[0, 90], [0, 45, -30, 90]])
def test_free_thermal_deformation_matches_laminate_theory(cfrp, stack):
    lam = Laminate.from_angles(cfrp, stack, 0.25e-3)
    state = HygrothermalState(-100.0, 0.002)
    mesh, bcs = free_plate(lam)
    result = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().set_hygrothermal(state))
    expected = np.linalg.solve(integrate_abd(lam), HygrothermalEngine().resultant_vector(lam, state))
    recoverer = ExactStressRecoverer()
    for element in mesh.elements:
        u = result.element_displacements(element.element_id)
        eps0, kappa, gamma = element.generalized_strains(u, 0.3, -0.2)
        assert np.allclose(np.concatenate([eps0, kappa]), expected, rtol=1e-6, atol=1e-8 * np.abs(expected).max())
        assert np.allclose(gamma, 0.0, atol=1e-10)
        # Free expansion: mechanical stresses are self-equilibrated through the thickness.
        N, M = np.zeros(3), np.zeros(3)
        for zb, zt in lam.ply_bounds():
            mid, half = 0.5 * (zb + zt), 0.5 * (zt - zb)
            for g in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
                z = mid + half * g
                sigma = recoverer.recover(element, u, z, state=state).stress_global
                N += sigma * half
                M += sigma * z * half
        scale = np.max(np.abs(lam.plies[0].Q_bar)) * 1e-3 * lam.thickness
        assert np.allclose(N, 0.0, atol=1e-6 * scale) and np.allclose(M, 0.0, atol=1e-6 * scale * lam.thickness)


def test_isotropic_free_expansion_is_stress_free(aluminium):
    lam = Laminate.from_angles(aluminium, [0, 0], 1e-3)
    state = HygrothermalState(150.0)
    mesh, bcs = free_plate(lam)
    result = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().set_hygrothermal(state))
    element = mesh.elements[5]
    for point in ExactStressRecoverer().through_thickness_profile(element, result.element_displacements(element.element_id), state=state):
        assert np.allclose(point.stress_global, 0.0, atol=1e-3)
        assert np.allclose(point.strain_total, point.strain_hygrothermal, rtol=1e-8)


def test_fully_restrained_cooldown_stress(aluminium):
    lam = Laminate.from_angles(aluminium, [0], 1e-3)
    mesh = FEAMesh.rectangular_plate(0.1, 0.1, 2, 2, lam, FSDTModel())
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UX, DOF.UY, DOF.UZ])
    state = HygrothermalState(-50.0)
    result = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().set_hygrothermal(state))
    point = ExactStressRecoverer().recover(mesh.elements[0], result.element_displacements(1), 0.0, state=state)
    E, nu, alpha = 68.9e9, 0.33, 23.6e-6
    expected = -E * alpha * (-50.0) / (1 - nu)
    assert np.allclose(point.stress_global[:2], expected, rtol=1e-9)
