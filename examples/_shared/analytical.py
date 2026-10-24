"""Closed-form Navier solutions for simply supported cross-ply plates (Reddy, 2004)."""

import numpy as np


def navier_clpt_centre(D, a, b, q, terms=61):
    """Centre deflection under uniform pressure q, classical (Kirchhoff) plate theory."""
    w = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            al, be = m * np.pi / a, n * np.pi / b
            denom = D[0, 0] * al**4 + 2 * (D[0, 1] + 2 * D[2, 2]) * al**2 * be**2 + D[1, 1] * be**4
            w += 16 * q / (np.pi**2 * m * n) / denom * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
    return w


def navier_clpt_centre_moments(D, a, b, q, terms=199):
    """Centre bending moments [Mx, My] under uniform pressure q (classical plate theory)."""
    Mx = My = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            al, be = m * np.pi / a, n * np.pi / b
            W = 16 * q / (np.pi**2 * m * n) / (D[0, 0] * al**4 + 2 * (D[0, 1] + 2 * D[2, 2]) * al**2 * be**2 + D[1, 1] * be**4)
            s = np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
            Mx += (D[0, 0] * al**2 + D[0, 1] * be**2) * W * s
            My += (D[0, 1] * al**2 + D[1, 1] * be**2) * W * s
    return np.array([Mx, My])


def navier_fsdt_centre(D, As, a, b, q, terms=61):
    """Centre deflection under uniform pressure q, first-order shear deformation theory.

    ``As`` is the 2x2 transverse shear stiffness ordered [yz, xz].
    """
    A44, A55 = As[0, 0], As[1, 1]
    w = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            al, be = m * np.pi / a, n * np.pi / b
            K = np.array(
                [
                    [A55 * al**2 + A44 * be**2, A55 * al, A44 * be],
                    [A55 * al, D[0, 0] * al**2 + D[2, 2] * be**2 + A55, (D[0, 1] + D[2, 2]) * al * be],
                    [A44 * be, (D[0, 1] + D[2, 2]) * al * be, D[2, 2] * al**2 + D[1, 1] * be**2 + A44],
                ]
            )
            W = np.linalg.solve(K, [16 * q / (np.pi**2 * m * n), 0.0, 0.0])[0]
            w += W * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
    return w


def navier_clpt_moment_field(D, a, b, q, x, y, terms=99):
    """Bending moments [Mx, My] at points (x, y) under uniform pressure q (classical plate theory)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    Mx, My = np.zeros_like(x), np.zeros_like(x)
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            al, be = m * np.pi / a, n * np.pi / b
            W = 16 * q / (np.pi**2 * m * n) / (D[0, 0] * al**4 + 2 * (D[0, 1] + 2 * D[2, 2]) * al**2 * be**2 + D[1, 1] * be**4)
            s = np.sin(al * x) * np.sin(be * y)
            Mx += (D[0, 0] * al**2 + D[0, 1] * be**2) * W * s
            My += (D[0, 1] * al**2 + D[1, 1] * be**2) * W * s
    return np.stack([Mx, My], axis=-1)
