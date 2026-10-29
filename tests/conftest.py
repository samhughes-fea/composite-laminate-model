import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from composite_laminate_model import (  # noqa: E402
    DOF,
    BoundaryConditions,
    EnrichedDOF,
    FEAMesh,
    Laminate,
    create_default_library,
)


@pytest.fixture(scope="session")
def library():
    return create_default_library()


@pytest.fixture(scope="session")
def cfrp(library):
    return library.get("T300/5208")


@pytest.fixture(scope="session")
def aluminium(library):
    return library.get("Al-6061-T6")


def simply_supported(mesh: FEAMesh, a: float, b: float, hard: bool = True) -> BoundaryConditions:
    """w = 0 on all edges plus in-plane rigid-body restraint.

    Hard SS also fixes the tangential rotation and the tangential enrichment amplitudes
    (zig-zag / interface displacements along the edge), which is the Navier condition.
    """
    bcs = BoundaryConditions()
    for n in mesh.nodes_on_line(0, 0.0) + mesh.nodes_on_line(0, a):
        bcs.fix([n], [DOF.UZ] + ([DOF.RX, EnrichedDOF.Y] if hard else []))
    for n in mesh.nodes_on_line(1, 0.0) + mesh.nodes_on_line(1, b):
        bcs.fix([n], [DOF.UZ] + ([DOF.RY, EnrichedDOF.X] if hard else []))
    bcs.fix([mesh.closest_node((0.0, 0.0))], [DOF.UX, DOF.UY])
    bcs.fix([mesh.closest_node((a, 0.0))], [DOF.UY])
    return bcs


def navier_clpt_centre(D: np.ndarray, a: float, b: float, q: float, terms: int = 61) -> float:
    """Centre deflection of a simply supported specially orthotropic plate (Navier series)."""
    w = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            denom = np.pi**4 * (
                D[0, 0] * (m / a) ** 4 + 2 * (D[0, 1] + 2 * D[2, 2]) * (m / a) ** 2 * (n / b) ** 2 + D[1, 1] * (n / b) ** 4
            )
            w += 16 * q / (np.pi**2 * m * n) / denom * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
    return w


def navier_fsdt_centre(D: np.ndarray, As: np.ndarray, a: float, b: float, q: float, terms: int = 61) -> float:
    """Centre deflection of a simply supported cross-ply plate in FSDT (Navier, Reddy 2004 sec. 6.3).

    Unknown amplitudes (W, X, Y) with w = W sin sin, beta_x = X cos sin, beta_y = Y sin cos.
    As is ordered [yz, xz] so A44 = As[0, 0], A55 = As[1, 1].
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
            qmn = 16 * q / (np.pi**2 * m * n)
            W = np.linalg.solve(K, np.array([qmn, 0.0, 0.0]))[0]
            w += W * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
    return w


def navier_clpt_centre_moments(D: np.ndarray, a: float, b: float, q: float, terms: int = 199) -> np.ndarray:
    """Centre bending moments [Mx, My] of the simply supported specially orthotropic plate (Navier)."""
    Mx = My = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            al, be = m * np.pi / a, n * np.pi / b
            W = 16 * q / (np.pi**2 * m * n) / (D[0, 0] * al**4 + 2 * (D[0, 1] + 2 * D[2, 2]) * al**2 * be**2 + D[1, 1] * be**4)
            s = np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
            # kappa = -w,xx etc. for w = W sin sin  ->  kx = al^2 W s, ky = be^2 W s
            Mx += (D[0, 0] * al**2 + D[0, 1] * be**2) * W * s
            My += (D[0, 1] * al**2 + D[1, 1] * be**2) * W * s
    return np.array([Mx, My])
