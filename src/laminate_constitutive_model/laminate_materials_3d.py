"""Three-dimensional orthotropic material property containers and tensor rotations.

Conventions used throughout the package
---------------------------------------
* Material axes: 1 = fibre direction, 2 = in-plane transverse, 3 = through-thickness.
* Voigt ordering of 3D stress / engineering-strain vectors: [11, 22, 33, 23, 13, 12].
* Rotations are about the 3 (= global z) axis by the ply angle theta (degrees),
  measured counter-clockwise from the global x axis to the fibre direction.
* All quantities are unit-agnostic; the default material library uses SI (Pa, m, K).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional

import numpy as np

VOIGT_LABELS_3D = ("11", "22", "33", "23", "13", "12")


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")


@dataclass(frozen=True)
class ElasticProperties3D:
    """Orthotropic elastic constants in principal material axes.

    ``v12``, ``v13`` and ``v23`` are the major Poisson ratios (strain in the second
    direction caused by stress in the first). The minor ratios follow from the
    reciprocity relation ``v_ji = v_ij * E_j / E_i``.
    """

    E1: float
    E2: float
    E3: float
    v12: float
    v23: float
    v13: float
    G12: float
    G23: float
    G13: float

    def __post_init__(self) -> None:
        for name in ("E1", "E2", "E3", "G12", "G23", "G13"):
            _require_positive(name, getattr(self, name))
        for name in ("v12", "v23", "v13"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        eigenvalues = np.linalg.eigvalsh(self.compliance_matrix())
        if np.min(eigenvalues) <= 0.0:
            raise ValueError(
                "Elastic constants are not thermodynamically admissible: the 3D "
                "compliance matrix is not positive definite."
            )

    # ------------------------------------------------------------------ factories
    @classmethod
    def isotropic(cls, E: float, nu: float) -> "ElasticProperties3D":
        G = E / (2.0 * (1.0 + nu))
        return cls(E, E, E, nu, nu, nu, G, G, G)

    @classmethod
    def transversely_isotropic(
        cls, E1: float, E2: float, v12: float, G12: float, v23: float
    ) -> "ElasticProperties3D":
        """Transversely isotropic about the fibre axis (2-3 plane of isotropy)."""
        G23 = E2 / (2.0 * (1.0 + v23))
        return cls(E1, E2, E2, v12, v23, v12, G12, G23, G12)

    # ----------------------------------------------------------- derived values
    @property
    def v21(self) -> float:
        return self.v12 * self.E2 / self.E1

    @property
    def v31(self) -> float:
        return self.v13 * self.E3 / self.E1

    @property
    def v32(self) -> float:
        return self.v23 * self.E3 / self.E2

    def compliance_matrix(self) -> np.ndarray:
        """6x6 compliance matrix S (strain = S @ stress) in Voigt order 11,22,33,23,13,12."""
        S = np.zeros((6, 6))
        S[0, 0] = 1.0 / self.E1
        S[1, 1] = 1.0 / self.E2
        S[2, 2] = 1.0 / self.E3
        S[0, 1] = S[1, 0] = -self.v12 / self.E1
        S[0, 2] = S[2, 0] = -self.v13 / self.E1
        S[1, 2] = S[2, 1] = -self.v23 / self.E2
        S[3, 3] = 1.0 / self.G23
        S[4, 4] = 1.0 / self.G13
        S[5, 5] = 1.0 / self.G12
        return S

    def stiffness_matrix(self) -> np.ndarray:
        """6x6 stiffness matrix C (stress = C @ strain)."""
        return np.linalg.inv(self.compliance_matrix())

    def reduced_stiffness(self) -> np.ndarray:
        """3x3 plane-stress reduced stiffness Q in order [11, 22, 12]."""
        denominator = 1.0 - self.v12 * self.v21
        Q11 = self.E1 / denominator
        Q22 = self.E2 / denominator
        Q12 = self.v12 * self.E2 / denominator
        return np.array(
            [[Q11, Q12, 0.0], [Q12, Q22, 0.0], [0.0, 0.0, self.G12]], dtype=float
        )

    def transverse_shear_stiffness(self) -> np.ndarray:
        """2x2 out-of-plane shear stiffness Q_s in order [23, 13]."""
        return np.array([[self.G23, 0.0], [0.0, self.G13]], dtype=float)

    def with_updates(self, **changes: float) -> "ElasticProperties3D":
        return replace(self, **changes)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class StrengthProperties3D:
    """Lamina strength allowables in principal material axes (all positive magnitudes).

    ``Zt``/``Zc`` are through-thickness normal strengths; ``S13``/``S23`` are the
    interlaminar (transverse) shear strengths. When omitted, the through-thickness
    values default to the transversely isotropic assumption (Zt=Yt, Zc=Yc, S13=S12)
    and S23 defaults to S12.
    """

    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S12: float
    Zt: Optional[float] = None
    Zc: Optional[float] = None
    S13: Optional[float] = None
    S23: Optional[float] = None

    def __post_init__(self) -> None:
        defaults = {"Zt": self.Yt, "Zc": self.Yc, "S13": self.S12, "S23": self.S12}
        for name, default in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, float(default))
        for f in fields(self):
            _require_positive(f.name, getattr(self, f.name))
            object.__setattr__(self, f.name, float(getattr(self, f.name)))

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------------------
# 3D rotations about the thickness axis (Bond matrices)
# ---------------------------------------------------------------------------
def reuter_matrix_3d() -> np.ndarray:
    """Reuter matrix converting tensor to engineering shear strains (Voigt order)."""
    return np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def stress_rotation_3d(theta_deg: float) -> np.ndarray:
    """Bond matrix T_sigma mapping global stresses to material axes: sigma_1 = T @ sigma_x.

    Voigt ordering [11, 22, 33, 23, 13, 12] with rotation theta about the 3-axis.
    """
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [
            [c * c, s * s, 0.0, 0.0, 0.0, 2.0 * c * s],
            [s * s, c * c, 0.0, 0.0, 0.0, -2.0 * c * s],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c, -s, 0.0],
            [0.0, 0.0, 0.0, s, c, 0.0],
            [-c * s, c * s, 0.0, 0.0, 0.0, c * c - s * s],
        ]
    )


def strain_rotation_3d(theta_deg: float) -> np.ndarray:
    """Engineering-strain rotation T_eps = R T_sigma R^-1 (eps_1 = T_eps @ eps_x)."""
    R = reuter_matrix_3d()
    return R @ stress_rotation_3d(theta_deg) @ np.linalg.inv(R)


def rotate_stiffness_3d(C: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate a 6x6 material stiffness into global axes: C_bar = T_sigma^-1 C T_eps."""
    return np.linalg.inv(stress_rotation_3d(theta_deg)) @ C @ strain_rotation_3d(theta_deg)
