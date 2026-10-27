"""Exact 3D elasticity solution for simply supported cross-ply plates (Pagano 1970).

A rectangular plate a x b, simply supported on all four edges (diaphragm supports),
carries a doubly sinusoidal transverse load q(x, y) = q0 sin(p x) sin(q y) on its top
face, with p = m pi / a and q = n pi / b. For orthotropic plies at 0 or 90 degrees
the displacement field

    u = U(z) cos(p x) sin(q y),  v = V(z) sin(p x) cos(q y),  w = W(z) sin(p x) sin(q y)

satisfies the edge conditions exactly, and the 3D equilibrium equations reduce to a
first-order system X' = A_k X for X = [U, V, W, Tx, Ty, Z] where Tx, Ty and Z are the
amplitudes of tau_xz, tau_yz and sigma_z. The transfer matrix exp(A_k h_k) carries the
state across each ply (split into sub-layers for conditioning); the six face
conditions tau_xz = tau_yz = 0 on both faces, sigma_z = 0 at the bottom and
sigma_z = q0 at the top fix the three unknown bottom displacements.

This is the reference solution used to rank the plate theories: no kinematic
assumption is made through the thickness, so sigma_z, interlaminar shear and the
transverse normal strain are all exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .laminate_materials_3d import rotate_stiffness_3d
from .laminate_mechanics import Laminate


def _expm(M: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring with a Pade(13) - or SciPy when available."""
    try:
        from scipy.linalg import expm

        return expm(M)
    except ImportError:  # pragma: no cover - SciPy is an optional dependency
        norm = np.linalg.norm(M, 1)
        s = max(0, int(np.ceil(np.log2(norm))) + 1) if norm > 0.5 else 0
        A = M / 2.0**s
        result, term = np.eye(len(M)), np.eye(len(M))
        for k in range(1, 30):
            term = term @ A / k
            result = result + term
        for _ in range(s):
            result = result @ result
        return result


@dataclass(frozen=True)
class PaganoProfile:
    """Stress and displacement amplitudes through the thickness (bottom to top).

    Multiply by the in-plane mode shapes to get field values:
    sigma_x, sigma_y, sigma_z, w ~ sin sin; tau_xz, u ~ cos sin; tau_yz, v ~ sin cos;
    tau_xy ~ cos cos.
    """

    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    sigma_x: np.ndarray
    sigma_y: np.ndarray
    sigma_z: np.ndarray
    tau_xz: np.ndarray
    tau_yz: np.ndarray
    tau_xy: np.ndarray
    ply_index: np.ndarray


class PaganoPlate:
    """Exact 3D solution of a simply supported cross-ply laminate under sinusoidal load."""

    def __init__(
        self,
        laminate: Laminate,
        a: float,
        b: float,
        q0: float = 1.0,
        m: int = 1,
        n: int = 1,
        sublayers: int = 8,
    ):
        for ply in laminate.plies:
            if min(abs(ply.angle % 90.0), abs(90.0 - ply.angle % 90.0)) > 1e-9:
                raise ValueError("Pagano's solution needs cross-ply laminates (plies at 0 or 90 degrees)")
        self.laminate = laminate
        self.a, self.b, self.q0 = float(a), float(b), float(q0)
        self.p, self.q = m * np.pi / a, n * np.pi / b
        self.sublayers = int(sublayers)
        self._C = [rotate_stiffness_3d(ply.material.elastic.stiffness_matrix(), ply.angle) for ply in laminate.plies]
        self._bottom = self._solve_bottom_state()

    # Voigt order of the stiffness: [11, 22, 33, 23, 13, 12]
    def _system(self, k: int) -> np.ndarray:
        C = self._C[k]
        p, q = self.p, self.q
        C11, C12, C13, C22, C23, C33 = C[0, 0], C[0, 1], C[0, 2], C[1, 1], C[1, 2], C[2, 2]
        C44, C55, C66 = C[3, 3], C[4, 4], C[5, 5]
        A = np.zeros((6, 6))
        # U' = Tx / C55 - p W ;  V' = Ty / C44 - q W
        A[0, 3], A[0, 2] = 1.0 / C55, -p
        A[1, 4], A[1, 2] = 1.0 / C44, -q
        # W' = (Z + p C13 U + q C23 V) / C33
        A[2, 5], A[2, 0], A[2, 1] = 1.0 / C33, p * C13 / C33, q * C23 / C33
        # Tx' = (p^2 C11 + q^2 C66) U + pq (C12 + C66) V - p C13 W'
        A[3, 0] = p * p * C11 + q * q * C66
        A[3, 1] = p * q * (C12 + C66)
        A[3] -= p * C13 * A[2]
        # Ty' = pq (C12 + C66) U + (p^2 C66 + q^2 C22) V - q C23 W'
        A[4, 0] = p * q * (C12 + C66)
        A[4, 1] = p * p * C66 + q * q * C22
        A[4] -= q * C23 * A[2]
        # Z' = p Tx + q Ty
        A[5, 3], A[5, 4] = p, q
        return A

    def _segments(self) -> List[Tuple[float, float, int]]:
        out = []
        for k, (z_b, z_t) in enumerate(self.laminate.ply_bounds()):
            edges = np.linspace(z_b, z_t, self.sublayers + 1)
            out += [(float(z0), float(z1), k) for z0, z1 in zip(edges[:-1], edges[1:])]
        return out

    def _solve_bottom_state(self) -> np.ndarray:
        # propagate the three unit bottom displacement states (tractions zero at the bottom)
        P = np.eye(6)
        for z0, z1, k in self._segments():
            P = _expm(self._system(k) * (z1 - z0)) @ P
        M = P[3:, :3]  # top tractions from bottom displacements
        U0 = np.linalg.solve(M, np.array([0.0, 0.0, self.q0]))
        return np.concatenate([U0, np.zeros(3)])

    def state_at(self, z: float) -> Tuple[np.ndarray, int]:
        X = self._bottom.copy()
        for z0, z1, k in self._segments():
            if z <= z1 + 1e-15:
                return _expm(self._system(k) * (z - z0)) @ X, k
            X = _expm(self._system(k) * (z1 - z0)) @ X
        return X, self.laminate.n_plies - 1

    def profile(self, points_per_ply: int = 21) -> PaganoProfile:
        rows, plies, zs = [], [], []
        p, q = self.p, self.q
        for k, (z_b, z_t) in enumerate(self.laminate.ply_bounds()):
            for z in np.linspace(z_b, z_t, points_per_ply):
                X, _ = self.state_at(float(z))
                U, V, W, Tx, Ty, Z = X
                C = self._C[k]
                Wp = (Z + p * C[0, 2] * U + q * C[1, 2] * V) / C[2, 2]
                sx = -p * C[0, 0] * U - q * C[0, 1] * V + C[0, 2] * Wp
                sy = -p * C[0, 1] * U - q * C[1, 1] * V + C[1, 2] * Wp
                txy = C[5, 5] * (q * U + p * V)
                rows.append((U, V, W, sx, sy, Z, Tx, Ty, txy))
                plies.append(k)
                zs.append(z)
        a = np.array(rows)
        return PaganoProfile(np.array(zs), *(a[:, i] for i in range(9)), np.array(plies))

    # ----------------------------------------------------------------- fields
    def centre_deflection(self) -> float:
        """w at (a/2, b/2) on the mid-plane."""
        return float(self.state_at(self.laminate.mid_plane)[0][2])

    def nondimensional(self) -> Dict[str, float]:
        """Pagano's normalised values for m = n = 1 with S = a / h.

        w = 100 E2 h^3 w(a/2, b/2, 0) / (q0 a^4) uses E2 of the first ply;
        sigma_x, sigma_y at (a/2, b/2) on the top face (sigma_y at the top of the second
        ply from the top for three-ply laminates follows the convention of reporting
        the faces); tau_xz at (0, b/2, 0) and tau_yz at (a/2, 0, 0).
        """
        lam = self.laminate
        h = lam.thickness
        S = self.a / h
        E2 = lam.plies[0].material.elastic.E2
        prof = self.profile(41)
        top = -1
        mid = int(np.argmin(np.abs(prof.z - lam.mid_plane)))
        return {
            "w": 100.0 * E2 * h**3 * self.centre_deflection() / (self.q0 * self.a**4),
            "sigma_x_top": prof.sigma_x[top] / (self.q0 * S * S),
            "sigma_x_bottom": prof.sigma_x[0] / (self.q0 * S * S),
            "sigma_y_top": prof.sigma_y[top] / (self.q0 * S * S),
            "tau_xz_mid": prof.tau_xz[mid] / (self.q0 * S),
            "tau_yz_mid": prof.tau_yz[mid] / (self.q0 * S),
        }
