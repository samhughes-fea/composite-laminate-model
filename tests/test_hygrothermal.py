"""Hygrothermal properties, expansion coefficients and resultants (the FE solves are in shell-fem)."""

import numpy as np
import pytest

trapezoid = getattr(np, "trapezoid", None) or np.trapz

from laminate_constitutive_model import (
    HygrothermalEngine,
    HygrothermalProperties,
    HygrothermalState,
    Laminate,
    Ply,
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
