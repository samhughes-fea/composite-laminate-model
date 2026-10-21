"""Ply and laminate geometry containers with plane-stress tensor transformations.

In-plane vectors use the order [x, y, xy] (engineering shear strain); transverse
shear vectors use [yz, xz] globally and [23, 13] in material axes, so that the
transverse shear stiffness is Q_s = diag(G23, G13).

The laminate z axis has its origin on the mid-plane; plies are listed from the
bottom (z = -h/2) to the top (z = +h/2).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .laminate_materials import MaterialCard


# ---------------------------------------------------------------------------
# Transformation matrices
# ---------------------------------------------------------------------------
def reuter_matrix() -> np.ndarray:
    """Reuter matrix R = diag(1, 1, 2): engineering strain = R @ tensor strain."""
    return np.diag([1.0, 1.0, 2.0])


def stress_transformation(theta_deg: float) -> np.ndarray:
    """In-plane stress transformation T: {s1, s2, t12} = T {sx, sy, txy}."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [
            [c * c, s * s, 2.0 * c * s],
            [s * s, c * c, -2.0 * c * s],
            [-c * s, c * s, c * c - s * s],
        ]
    )


def strain_transformation(theta_deg: float) -> np.ndarray:
    """Engineering strain transformation R T R^-1: {e1, e2, g12} = T_eps {ex, ey, gxy}."""
    R = reuter_matrix()
    return R @ stress_transformation(theta_deg) @ np.linalg.inv(R)


def shear_transformation(theta_deg: float) -> np.ndarray:
    """Transverse shear rotation T_s: {g23, g13} = T_s {g_yz, g_xz} (same for stresses)."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def transform_reduced_stiffness(Q: np.ndarray, theta_deg: float) -> np.ndarray:
    """Q_bar = T^-1 Q R T R^-1 (rotation of the plane-stress stiffness to global axes)."""
    T = stress_transformation(theta_deg)
    R = reuter_matrix()
    return np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)


def transform_shear_stiffness(Q_s: np.ndarray, theta_deg: float) -> np.ndarray:
    """Q_bar_s = T_s^T Q_s T_s (T_s is orthogonal)."""
    T_s = shear_transformation(theta_deg)
    return T_s.T @ Q_s @ T_s


def stiffness_invariants(Q: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Tsai-Pagano invariants U1..U5 of a plane-stress orthotropic stiffness."""
    Q11, Q22, Q12, Q66 = Q[0, 0], Q[1, 1], Q[0, 1], Q[2, 2]
    U1 = (3.0 * Q11 + 3.0 * Q22 + 2.0 * Q12 + 4.0 * Q66) / 8.0
    U2 = (Q11 - Q22) / 2.0
    U3 = (Q11 + Q22 - 2.0 * Q12 - 4.0 * Q66) / 8.0
    U4 = (Q11 + Q22 + 6.0 * Q12 - 4.0 * Q66) / 8.0
    U5 = (Q11 + Q22 - 2.0 * Q12 + 4.0 * Q66) / 8.0
    return U1, U2, U3, U4, U5


def q_bar_from_invariants(Q: np.ndarray, theta_deg: float) -> np.ndarray:
    """Q_bar assembled from the invariant (multiple-angle) form; equals transform_reduced_stiffness."""
    U1, U2, U3, U4, U5 = stiffness_invariants(Q)
    t = np.radians(theta_deg)
    c2, s2, c4, s4 = np.cos(2 * t), np.sin(2 * t), np.cos(4 * t), np.sin(4 * t)
    Qb11 = U1 + U2 * c2 + U3 * c4
    Qb22 = U1 - U2 * c2 + U3 * c4
    Qb12 = U4 - U3 * c4
    Qb66 = U5 - U3 * c4
    Qb16 = 0.5 * U2 * s2 + U3 * s4
    Qb26 = 0.5 * U2 * s2 - U3 * s4
    return np.array([[Qb11, Qb12, Qb16], [Qb12, Qb22, Qb26], [Qb16, Qb26, Qb66]])


# ---------------------------------------------------------------------------
# Ply
# ---------------------------------------------------------------------------
class Ply:
    """Single lamina; stiffness matrices are precomputed at construction.

    Attributes (read-only): ``Q``, ``Q_s`` (material axes), ``Q_bar``, ``Q_bar_s``
    (global shell axes), ``T`` (stress rotation), ``T_s`` (shear rotation).
    """

    __slots__ = ("_material", "_thickness", "_angle", "_name", "_Q", "_Q_s", "_Q_bar", "_Q_bar_s", "_T", "_T_s")

    def __init__(self, material: MaterialCard, thickness: float, angle: float = 0.0, name: Optional[str] = None):
        if not isinstance(material, MaterialCard):
            raise TypeError("Ply.material must be a MaterialCard")
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("Ply thickness must be positive")
        self._material = material
        self._thickness = float(thickness)
        self._angle = float(angle)
        self._name = name or f"{material.name}@{angle:g}"
        self._Q = material.elastic.reduced_stiffness()
        self._Q_s = material.elastic.transverse_shear_stiffness()
        self._T = stress_transformation(self._angle)
        self._T_s = shear_transformation(self._angle)
        self._Q_bar = transform_reduced_stiffness(self._Q, self._angle)
        self._Q_bar_s = transform_shear_stiffness(self._Q_s, self._angle)
        for array in (self._Q, self._Q_s, self._T, self._T_s, self._Q_bar, self._Q_bar_s):
            array.setflags(write=False)

    # -------------------------------------------------------------- properties
    material = property(lambda self: self._material)
    thickness = property(lambda self: self._thickness)
    angle = property(lambda self: self._angle)
    name = property(lambda self: self._name)
    Q = property(lambda self: self._Q)
    Q_s = property(lambda self: self._Q_s)
    Q_bar = property(lambda self: self._Q_bar)
    Q_bar_s = property(lambda self: self._Q_bar_s)
    T = property(lambda self: self._T)
    T_s = property(lambda self: self._T_s)

    @property
    def T_strain(self) -> np.ndarray:
        return strain_transformation(self._angle)

    def with_material(self, material: MaterialCard) -> "Ply":
        """New ply with the same geometry and a different material (e.g. degraded)."""
        return Ply(material, self._thickness, self._angle, self._name)

    def to_material_stress(self, sigma_global: np.ndarray) -> np.ndarray:
        return self._T @ np.asarray(sigma_global, dtype=float)

    def to_material_strain(self, eps_global: np.ndarray) -> np.ndarray:
        return self.T_strain @ np.asarray(eps_global, dtype=float)

    def to_material_shear(self, tau_global: np.ndarray) -> np.ndarray:
        return self._T_s @ np.asarray(tau_global, dtype=float)

    def __repr__(self) -> str:
        return f"Ply({self._material.name!r}, t={self._thickness:g}, angle={self._angle:g})"


# ---------------------------------------------------------------------------
# Laminate
# ---------------------------------------------------------------------------
class Laminate:
    """Immutable ordered stack of plies, bottom to top, centred on the mid-plane."""

    __slots__ = ("_plies", "_name", "_z")

    def __init__(self, plies: Sequence[Ply], name: str = ""):
        plies = tuple(plies)
        if not plies:
            raise ValueError("A laminate needs at least one ply")
        if not all(isinstance(p, Ply) for p in plies):
            raise TypeError("Laminate plies must be Ply instances")
        self._plies = plies
        self._name = name
        thicknesses = np.array([p.thickness for p in plies])
        z = np.concatenate([[0.0], np.cumsum(thicknesses)]) - 0.5 * thicknesses.sum()
        z.setflags(write=False)
        self._z = z

    @classmethod
    def from_angles(
        cls, material: MaterialCard, angles: Iterable[float], ply_thickness: float, name: str = ""
    ) -> "Laminate":
        angles = list(angles)
        return cls([Ply(material, ply_thickness, a) for a in angles], name or cls._sequence(angles))

    @staticmethod
    def _sequence(angles: Sequence[float]) -> str:
        return "[" + "/".join(f"{a:g}" for a in angles) + "]"

    plies = property(lambda self: self._plies)
    name = property(lambda self: self._name or self.stacking_sequence)

    @property
    def stacking_sequence(self) -> str:
        return self._sequence([p.angle for p in self._plies])

    @property
    def n_plies(self) -> int:
        return len(self._plies)

    @property
    def thickness(self) -> float:
        return float(self._z[-1] - self._z[0])

    @property
    def z_interfaces(self) -> np.ndarray:
        """N+1 interface coordinates from bottom to top."""
        return self._z

    def ply_bounds(self) -> List[Tuple[float, float]]:
        return [(float(self._z[k]), float(self._z[k + 1])) for k in range(self.n_plies)]

    def ply_midplanes(self) -> np.ndarray:
        return 0.5 * (self._z[:-1] + self._z[1:])

    def ply_index_at(self, z: float, side: str = "above") -> int:
        """Index of the ply containing z. At an interface, ``side`` picks the ply above or below."""
        tol = 1e-12 * max(self.thickness, 1.0)
        if z < self._z[0] - tol or z > self._z[-1] + tol:
            raise ValueError(f"z={z} lies outside the laminate [{self._z[0]}, {self._z[-1]}]")
        if side not in ("above", "below"):
            raise ValueError("side must be 'above' or 'below'")
        if side == "above":
            index = int(np.searchsorted(self._z, z + tol, side="right")) - 1
        else:
            index = int(np.searchsorted(self._z, z - tol, side="left")) - 1
        return int(np.clip(index, 0, self.n_plies - 1))

    def is_symmetric(self, tol: float = 1e-9) -> bool:
        n = self.n_plies
        for k in range(n // 2):
            a, b = self._plies[k], self._plies[n - 1 - k]
            if (
                abs(a.angle - b.angle) > tol
                or abs(a.thickness - b.thickness) > tol * max(a.thickness, 1.0)
                or a.material != b.material
            ):
                return False
        return True

    def is_balanced(self, tol: float = 1e-6) -> bool:
        """True when the extensional shear couplings vanish (A16 = A26 = 0 relative to A11)."""
        A = sum(p.Q_bar * p.thickness for p in self._plies)
        scale = abs(A[0, 0])
        return abs(A[0, 2]) <= tol * scale and abs(A[1, 2]) <= tol * scale

    def replace_ply(self, index: int, ply: Ply) -> "Laminate":
        plies = list(self._plies)
        plies[index] = ply
        return Laminate(plies, self._name)

    def areal_density(self) -> float:
        return float(sum(p.material.density * p.thickness for p in self._plies))

    def __len__(self) -> int:
        return self.n_plies

    def __iter__(self):
        return iter(self._plies)

    def __repr__(self) -> str:
        return f"Laminate({self.stacking_sequence}, h={self.thickness:g})"
