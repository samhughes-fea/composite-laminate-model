"""Lamina failure criteria (Strategy Pattern).

Every criterion consumes a material-axis stress vector in Voigt order
[s1, s2, s3, t23, t13, t12] and a ``StrengthProperties3D``, and returns a
``FailureResult``. ``index`` >= 1 means failure; ``strength_ratio`` is the factor
by which the (proportionally scaled) stress state can grow before failure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Union

import numpy as np

from .laminate_materials_3d import StrengthProperties3D


class FailureMode(Enum):
    NONE = "none"
    FIBER_TENSION = "fiber_tension"
    FIBER_COMPRESSION = "fiber_compression"
    MATRIX_TENSION = "matrix_tension"
    MATRIX_COMPRESSION = "matrix_compression"
    IN_PLANE_SHEAR = "in_plane_shear"
    TRANSVERSE_NORMAL = "transverse_normal"
    DELAMINATION = "delamination"
    INTERACTIVE = "interactive"


FAILURE_TOLERANCE = 1e-9  # indices within round-off of 1 count as failure


@dataclass(frozen=True)
class FailureResult:
    criterion: str
    index: float
    strength_ratio: float
    mode: FailureMode
    mode_indices: Dict[str, float] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.index >= 1.0 - FAILURE_TOLERANCE

    def active_modes(self) -> set:
        """All modes whose individual index reached 1."""
        return {FailureMode(name) for name, value in self.mode_indices.items() if value >= 1.0 - FAILURE_TOLERANCE}


def _as_stress(stress) -> np.ndarray:
    s = np.asarray(stress, dtype=float).ravel()
    if s.size == 3:  # plane stress [s1, s2, t12]
        s = np.array([s[0], s[1], 0.0, 0.0, 0.0, s[2]])
    if s.size != 6:
        raise ValueError("stress must have 3 (plane) or 6 (Voigt 3D) components")
    return s


def _quadratic_ratio(index: float) -> float:
    return float("inf") if index <= 0.0 else float(1.0 / np.sqrt(index))


def _mode_from_matrix_sign(s2: float) -> FailureMode:
    return FailureMode.MATRIX_TENSION if s2 >= 0.0 else FailureMode.MATRIX_COMPRESSION


class FailureCriterion(ABC):
    name: str = "criterion"

    @abstractmethod
    def evaluate(self, stress, strength: StrengthProperties3D) -> FailureResult:
        """Evaluate the criterion for a material-axis stress state."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MaximumStressCriterion(FailureCriterion):
    """Non-interactive maximum stress criterion in all six components."""

    name = "MaxStress"

    def evaluate(self, stress, strength):
        s1, s2, s3, t23, t13, t12 = _as_stress(stress)
        ratios = {
            FailureMode.FIBER_TENSION.value: max(s1, 0.0) / strength.Xt,
            FailureMode.FIBER_COMPRESSION.value: max(-s1, 0.0) / strength.Xc,
            FailureMode.MATRIX_TENSION.value: max(s2, 0.0) / strength.Yt,
            FailureMode.MATRIX_COMPRESSION.value: max(-s2, 0.0) / strength.Yc,
            FailureMode.TRANSVERSE_NORMAL.value: max(s3 / strength.Zt, -s3 / strength.Zc, 0.0),
            FailureMode.IN_PLANE_SHEAR.value: abs(t12) / strength.S12,
            FailureMode.DELAMINATION.value: max(abs(t13) / strength.S13, abs(t23) / strength.S23),
        }
        mode_name = max(ratios, key=ratios.get)
        index = ratios[mode_name]
        mode = FailureMode(mode_name) if index > 0.0 else FailureMode.NONE
        ratio = float("inf") if index == 0.0 else 1.0 / index
        return FailureResult(self.name, float(index), ratio, mode, ratios)


class TsaiHillCriterion(FailureCriterion):
    """Tsai-Hill criterion: Hill's orthotropic quadratic form with sign-dependent strengths.

    The normal-stress coefficients assume transverse isotropy (through-thickness
    normal strength equal to the governing transverse strength Y), which keeps the
    failure surface closed for any strength set:

        index = (s1^2 - s1 s2 - s1 s3)/X^2 + (s2^2 + s3^2)/Y^2 - (2/Y^2 - 1/X^2) s2 s3
                + (t23/S23)^2 + (t13/S13)^2 + (t12/S12)^2

    In plane stress this is (s1/X)^2 - s1 s2/X^2 + (s2/Y)^2 + (t12/S)^2.
    """

    name = "TsaiHill"

    def evaluate(self, stress, strength):
        s1, s2, s3, t23, t13, t12 = _as_stress(stress)
        X = strength.Xt if s1 >= 0.0 else strength.Xc
        Y = strength.Yt if s2 + s3 >= 0.0 else strength.Yc
        H = G = 0.5 / X**2
        F = 1.0 / Y**2 - 0.5 / X**2
        index = (
            F * (s2 - s3) ** 2
            + G * (s3 - s1) ** 2
            + H * (s1 - s2) ** 2
            + (t23 / strength.S23) ** 2
            + (t13 / strength.S13) ** 2
            + (t12 / strength.S12) ** 2
        )
        contributions = {
            "fiber": (s1 / X) ** 2,
            "matrix": (s2 / Y) ** 2,
            "shear": (t12 / strength.S12) ** 2,
            "interlaminar": (t13 / strength.S13) ** 2 + (t23 / strength.S23) ** 2,
        }
        mode = FailureMode.INTERACTIVE if index > 0.0 else FailureMode.NONE
        return FailureResult(self.name, float(index), _quadratic_ratio(index), mode, contributions)


class TsaiWuCriterion(FailureCriterion):
    """Tsai-Wu interactive tensor polynomial F_i s_i + F_ij s_i s_j = 1.

    ``interaction`` selects the normal-stress interaction terms:
    * ``"hoffman"`` (default): F12 = -1/(2 Xt Xc) in plane stress; in 3D the Hoffman
      values F12 = -C3, F13 = -C2, F23 = -C1 from C2+C3 = 1/(XtXc), C1+C3 = 1/(YtYc),
      C1+C2 = 1/(ZtZc).
    * ``"tsai-wu"``: F_ij = -1/2 sqrt(F_ii F_jj).
    * a float f*: F_ij = f* sqrt(F_ii F_jj) for every pair.
    """

    name = "TsaiWu"

    def __init__(self, interaction: Union[str, float] = "hoffman"):
        if isinstance(interaction, str) and interaction not in ("hoffman", "tsai-wu"):
            raise ValueError("interaction must be 'hoffman', 'tsai-wu' or a float")
        if not isinstance(interaction, str) and not -1.0 < float(interaction) < 1.0:
            raise ValueError("a numeric interaction coefficient must lie in (-1, 1)")
        self.interaction = interaction

    def coefficients(self, strength: StrengthProperties3D):
        Xt, Xc, Yt, Yc, Zt, Zc = strength.Xt, strength.Xc, strength.Yt, strength.Yc, strength.Zt, strength.Zc
        F = np.array([1 / Xt - 1 / Xc, 1 / Yt - 1 / Yc, 1 / Zt - 1 / Zc, 0.0, 0.0, 0.0])
        Fd = np.array([1 / (Xt * Xc), 1 / (Yt * Yc), 1 / (Zt * Zc),
                       1 / strength.S23**2, 1 / strength.S13**2, 1 / strength.S12**2])
        FF = np.diag(Fd)
        if self.interaction == "hoffman":
            C3 = 0.5 * (Fd[0] + Fd[1] - Fd[2])
            C2 = Fd[0] - C3
            C1 = Fd[1] - C3
            F12, F13, F23 = -C3, -C2, -C1
        else:
            f = -0.5 if self.interaction == "tsai-wu" else float(self.interaction)
            F12 = f * np.sqrt(Fd[0] * Fd[1])
            F13 = f * np.sqrt(Fd[0] * Fd[2])
            F23 = f * np.sqrt(Fd[1] * Fd[2])
        FF[0, 1] = FF[1, 0] = F12
        FF[0, 2] = FF[2, 0] = F13
        FF[1, 2] = FF[2, 1] = F23
        return F, FF

    def evaluate(self, stress, strength):
        s = _as_stress(stress)
        F, FF = self.coefficients(strength)
        a = float(s @ FF @ s)
        b = float(F @ s)
        index = a + b
        if a > 0.0:
            ratio = (-b + np.sqrt(b * b + 4.0 * a)) / (2.0 * a)
        elif b > 0.0:
            ratio = 1.0 / b
        else:
            ratio = float("inf")
        contributions = {
            "fiber": F[0] * s[0] + FF[0, 0] * s[0] ** 2,
            "matrix": F[1] * s[1] + FF[1, 1] * s[1] ** 2,
            "shear": FF[5, 5] * s[5] ** 2,
            "interlaminar": FF[3, 3] * s[3] ** 2 + FF[4, 4] * s[4] ** 2,
        }
        mode = FailureMode.INTERACTIVE if index > 0.0 else FailureMode.NONE
        return FailureResult(self.name, float(index), float(ratio), mode, contributions)


class HashinCriterion(FailureCriterion):
    """Hashin (1980) mode-separated criterion.

    ``variant="2d"``: plane-stress fibre/matrix tension/compression modes.
    ``variant="3d"``: full 3D fibre and matrix modes plus a quadratic interlaminar
    delamination mode (s3/Z)^2 + (t13/S13)^2 + (t23/S23)^2 driven by the
    through-thickness stresses (in a shell s3 = 0, so the interlaminar shears govern).
    Mode indices are quadratic; ``strength_ratio`` is the proportional-loading
    reserve of the governing mode.
    """

    def __init__(self, variant: str = "3d"):
        if variant not in ("2d", "3d"):
            raise ValueError("variant must be '2d' or '3d'")
        self.variant = variant
        self.name = f"Hashin{variant.upper()}"

    def _modes_2d(self, s, st) -> Dict[str, float]:
        s1, s2, _, _, _, t12 = s
        modes = {}
        if s1 >= 0.0:
            modes[FailureMode.FIBER_TENSION.value] = (s1 / st.Xt) ** 2 + (t12 / st.S12) ** 2
        else:
            modes[FailureMode.FIBER_COMPRESSION.value] = (s1 / st.Xc) ** 2
        if s2 >= 0.0:
            modes[FailureMode.MATRIX_TENSION.value] = (s2 / st.Yt) ** 2 + (t12 / st.S12) ** 2
        else:
            modes[FailureMode.MATRIX_COMPRESSION.value] = (
                (s2 / (2.0 * st.S23)) ** 2
                + ((st.Yc / (2.0 * st.S23)) ** 2 - 1.0) * s2 / st.Yc
                + (t12 / st.S12) ** 2
            )
        return modes

    def _modes_3d(self, s, st) -> Dict[str, float]:
        s1, s2, s3, t23, t13, t12 = s
        modes = {}
        long_shear = (t12**2 + t13**2) / st.S12**2
        if s1 >= 0.0:
            modes[FailureMode.FIBER_TENSION.value] = (s1 / st.Xt) ** 2 + long_shear
        else:
            modes[FailureMode.FIBER_COMPRESSION.value] = (s1 / st.Xc) ** 2
        s23 = s2 + s3
        trans_shear = (t23**2 - s2 * s3) / st.S23**2
        if s23 >= 0.0:
            modes[FailureMode.MATRIX_TENSION.value] = (s23 / st.Yt) ** 2 + trans_shear + long_shear
        else:
            modes[FailureMode.MATRIX_COMPRESSION.value] = (
                ((st.Yc / (2.0 * st.S23)) ** 2 - 1.0) * s23 / st.Yc
                + s23**2 / (4.0 * st.S23**2)
                + trans_shear
                + long_shear
            )
        Z = st.Zt if s3 >= 0.0 else st.Zc
        modes[FailureMode.DELAMINATION.value] = (s3 / Z) ** 2 + (t13 / st.S13) ** 2 + (t23 / st.S23) ** 2
        return modes

    def evaluate(self, stress, strength):
        s = _as_stress(stress)
        modes = self._modes_2d(s, strength) if self.variant == "2d" else self._modes_3d(s, strength)
        mode_name = max(modes, key=modes.get)
        index = float(modes[mode_name])
        mode = FailureMode(mode_name) if index > 0.0 else FailureMode.NONE
        ratio = self._strength_ratio(s, strength, mode_name) if index > 0.0 else float("inf")
        return FailureResult(self.name, index, ratio, mode, {k: float(v) for k, v in modes.items()})

    def _strength_ratio(self, s, strength, mode_name) -> float:
        """Smallest load factor at which any mode reaches 1, solving each mode's quadratic."""
        best = float("inf")
        evaluator = self._modes_2d if self.variant == "2d" else self._modes_3d
        m1 = evaluator(s, strength)
        m2 = evaluator(2.0 * s, strength)
        for name in m1:
            if name not in m2:
                continue
            # index(lambda) = a lambda^2 + b lambda, fitted exactly from lambda = 1 and 2.
            a = 0.5 * (m2[name] - 2.0 * m1[name])
            b = m1[name] - a
            if abs(a) < 1e-300:
                lam = 1.0 / b if b > 0.0 else float("inf")
            else:
                disc = b * b + 4.0 * a
                if disc < 0.0:
                    continue
                roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
                positive = [r for r in roots if r > 0.0]
                lam = min(positive) if positive else float("inf")
            best = min(best, lam)
        return float(best)


CRITERIA = {
    "max_stress": MaximumStressCriterion,
    "tsai_hill": TsaiHillCriterion,
    "tsai_wu": TsaiWuCriterion,
    "hashin_2d": lambda: HashinCriterion("2d"),
    "hashin_3d": lambda: HashinCriterion("3d"),
}


def create_criterion(name: str) -> FailureCriterion:
    try:
        return CRITERIA[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown failure criterion '{name}'. Options: {', '.join(CRITERIA)}") from exc
