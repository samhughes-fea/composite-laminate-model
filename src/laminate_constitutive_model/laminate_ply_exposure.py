"""Ply failure exposures for many stress states at once, including Puck's criterion.

The criteria in ``laminate_failure_criteria`` take one stress state and return a
``FailureResult``. This module evaluates whole arrays of states, as a beam or shell
stress recovery produces them, and returns **exposures**.

An exposure ``f`` is the factor by which a stress state is past (``f > 1``) or short of
(``f < 1``) the failure surface along a proportional load path, so ``f = 1 /
strength_ratio`` and it scales linearly with the load. The reserve factor is ``1 / f``
and the margin of safety ``1 / f - 1``.

Stresses are in material axes, Voigt order ``[s1, s2, s3, t23, t13, t12]``, shape
``(..., 6)``. Strengths come from ``StrengthProperties3D``; ``S13`` is used for ``t13``
and ``S23`` for ``t23``.

- **Maximum stress:** each stress against its strength.
- **Tsai-Wu:** the 3D tensor polynomial with the same interaction options as
  ``TsaiWuCriterion`` (``"hoffman"``, ``"tsai-wu"`` or a number).
- **Hashin (1980, 3D):** fibre and matrix tension and compression. Unlike
  ``HashinCriterion("3d")`` it has no separate delamination mode.
- **Puck** (Puck and Schuermann 2002; VDI 2014): fibre failure ``s1 / X``, and
  inter-fibre failure on the most stressed action plane, found by a search over the
  plane angle. In plane stress it reduces to Puck's modes A, B and C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np

from .laminate_materials_3d import StrengthProperties3D

__all__ = [
    "EXPOSURE_CRITERIA",
    "PuckParameters",
    "hashin_exposure",
    "max_stress_exposure",
    "ply_exposures",
    "puck_exposure",
    "tsai_wu_exposure",
]

#: criteria returned by ``ply_exposures``, in output order
EXPOSURE_CRITERIA = ("max_stress", "tsai_wu", "hashin", "puck")


@dataclass(frozen=True)
class PuckParameters:
    """Puck's inclination parameters ``p_perp_par^+``, ``p_perp_par^-`` and ``p_perp_perp``.

    The defaults (0.30, 0.25, 0.25) are glass-epoxy values; carbon-epoxy is about 0.35,
    0.30 and 0.25-0.30.
    """

    p_TL_t: float = 0.30
    p_TL_c: float = 0.25
    p_TT: float = 0.25

    def __post_init__(self) -> None:
        for name in ("p_TL_t", "p_TL_c", "p_TT"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"PuckParameters.{name} must lie in [0, 1), got {value}")

    def R_TT(self, strength: StrengthProperties3D) -> float:
        """Fracture resistance of the action plane against transverse shear, ``Yc / (2 (1 + p_TT))``."""
        return strength.Yc / (2.0 * (1.0 + self.p_TT))


def _stress(stress) -> np.ndarray:
    s = np.asarray(stress, dtype=np.float64)
    if s.shape[-1] != 6:
        raise ValueError(f"ply stresses must have 6 components (..., 6), got shape {s.shape}")
    return s


def _quadratic_exposure(a, b) -> np.ndarray:
    """``f = 1 / R`` with ``a R^2 + b R = 1`` and ``a >= 0``; ``f = 0`` for a zero state."""
    a = np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    b = np.asarray(b, dtype=np.float64)
    return 0.5 * (b + np.sqrt(b * b + 4.0 * a))


def max_stress_exposure(stress, strength: StrengthProperties3D) -> Tuple[np.ndarray, np.ndarray]:
    """Maximum-stress exposure and the governing component (``"s11t"``, ``"s22c"``, ``"t12"`` ...)."""
    s = _stress(stress)
    s1, s2, s3, t23, t13, t12 = np.moveaxis(s, -1, 0)
    st = strength
    terms = np.stack([
        np.where(s1 >= 0, s1 / st.Xt, -s1 / st.Xc),
        np.where(s2 >= 0, s2 / st.Yt, -s2 / st.Yc),
        np.where(s3 >= 0, s3 / st.Zt, -s3 / st.Zc),
        np.abs(t23) / st.S23,
        np.abs(t13) / st.S13,
        np.abs(t12) / st.S12,
    ])
    k = np.argmax(terms, axis=0)
    names = np.array([["s11t", "s11c"], ["s22t", "s22c"], ["s33t", "s33c"], ["t23", "t23"], ["t13", "t13"],
                      ["t12", "t12"]])
    component = np.take_along_axis(np.moveaxis(s, -1, 0), k[None], 0)[0]
    mode = names[k, (component < 0).astype(int)]
    return np.take_along_axis(terms, k[None], 0)[0], mode


def tsai_wu_exposure(stress, strength: StrengthProperties3D,
                     interaction: Union[str, float] = "hoffman") -> np.ndarray:
    """Tsai-Wu exposure, with the interaction terms of ``TsaiWuCriterion(interaction)``."""
    from .laminate_failure_criteria import TsaiWuCriterion

    s = _stress(stress)
    F, FF = TsaiWuCriterion(interaction).coefficients(strength)
    a = np.einsum("...i,ij,...j->...", s, FF, s)
    b = s @ F
    return _quadratic_exposure(a, b)


def hashin_exposure(stress, strength: StrengthProperties3D) -> Tuple[np.ndarray, np.ndarray]:
    """Hashin (1980, 3D) exposure and mode: ``"FT"``, ``"FC"``, ``"MT"`` or ``"MC"``."""
    s = _stress(stress)
    s1, s2, s3, t23, t13, t12 = np.moveaxis(s, -1, 0)
    st = strength
    long_shear = (t12 / st.S12) ** 2 + (t13 / st.S13) ** 2
    fibre = np.where(s1 >= 0, np.sqrt((s1 / st.Xt) ** 2 + long_shear), -s1 / st.Xc)
    sn = s2 + s3
    trans_shear = (t23**2 - s2 * s3) / st.S23**2
    tension = np.sqrt(np.maximum((sn / st.Yt) ** 2 + trans_shear + long_shear, 0.0))
    a_c = (sn / (2 * st.S23)) ** 2 + trans_shear + long_shear
    b_c = ((st.Yc / (2 * st.S23)) ** 2 - 1.0) * sn / st.Yc
    compression = _quadratic_exposure(a_c, b_c)
    matrix = np.where(sn >= 0, tension, compression)
    f = np.maximum(fibre, matrix)
    mode = np.where(fibre >= matrix, np.where(s1 >= 0, "FT", "FC"), np.where(sn >= 0, "MT", "MC"))
    return f, mode


_THETA = np.radians(np.arange(-90.0, 90.0, 1.0))


def _puck_iff(s2, s3, t23, t13, t12, strength, puck: PuckParameters, theta) -> np.ndarray:
    """Puck's inter-fibre exposure on the action plane at ``theta`` (broadcast)."""
    c, sn_ = np.cos(theta), np.sin(theta)
    sig_n = s2 * c * c + s3 * sn_ * sn_ + 2 * t23 * sn_ * c
    tau_nt = (s3 - s2) * sn_ * c + t23 * (c * c - sn_ * sn_)
    tau_n1 = t13 * sn_ + t12 * c
    RTp, RTT, RTL = strength.Yt, puck.R_TT(strength), strength.S12
    shear2 = tau_nt**2 + tau_n1**2
    with np.errstate(invalid="ignore", divide="ignore"):
        cos2 = np.where(shear2 > 0, tau_nt**2 / shear2, 0.0)
    sin2 = np.where(shear2 > 0, 1.0 - cos2, 1.0)
    pt = puck.p_TT / RTT * cos2 + puck.p_TL_t / RTL * sin2  # p_perp_psi^+ / R_perp_psi^A
    pc = puck.p_TT / RTT * cos2 + puck.p_TL_c / RTL * sin2  # p_perp_psi^- / R_perp_psi^A
    tension = np.sqrt(((1 / RTp - pt) * sig_n) ** 2 + (tau_nt / RTT) ** 2 + (tau_n1 / RTL) ** 2) + pt * sig_n
    compression = np.sqrt((tau_nt / RTT) ** 2 + (tau_n1 / RTL) ** 2 + (pc * sig_n) ** 2) + pc * sig_n
    return np.where(sig_n >= 0, tension, compression)


def puck_exposure(stress, strength: StrengthProperties3D,
                  puck: PuckParameters = PuckParameters()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Puck exposure, mode (``"FFT"``, ``"FFC"``, ``"IFF-A"``, ``"IFF-B"``, ``"IFF-C"``) and the
    fracture-plane angle in degrees.

    The action plane is searched in 1-degree steps over [-90, 90) and refined by a
    parabola through the best step and its neighbours. Modes A, B and C are named as in
    plane stress: A for a tensile normal stress on the fracture plane, B for a compressive
    one on the plane at 0 degrees, C on an inclined plane. Puck uses ``Yt``, ``Yc`` and
    ``S12`` as its action-plane resistances; ``S13`` and ``S23`` are not used.
    """
    s = _stress(stress)
    s1, s2, s3, t23, t13, t12 = (x[..., None] for x in np.moveaxis(s, -1, 0))
    th = _THETA
    f_all = _puck_iff(s2, s3, t23, t13, t12, strength, puck, th)
    k = np.argmax(f_all, axis=-1)
    n = th.size
    km, kp = (k - 1) % n, (k + 1) % n
    f0 = np.take_along_axis(f_all, k[..., None], -1)[..., 0]
    fm = np.take_along_axis(f_all, km[..., None], -1)[..., 0]
    fp = np.take_along_axis(f_all, kp[..., None], -1)[..., 0]
    den = fm - 2 * f0 + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(den < 0, 0.5 * (fm - fp) / den, 0.0)
    d = np.clip(d, -0.5, 0.5)
    th_fp = th[k] + d * (th[1] - th[0])
    s2, s3, t23, t13, t12, s1 = (x[..., 0] for x in (s2, s3, t23, t13, t12, s1))
    iff = np.maximum(_puck_iff(s2, s3, t23, t13, t12, strength, puck, th_fp), f0)
    ff = np.where(s1 >= 0, s1 / strength.Xt, -s1 / strength.Xc)
    c, sn_ = np.cos(th_fp), np.sin(th_fp)
    sig_n = s2 * c * c + s3 * sn_ * sn_ + 2 * t23 * sn_ * c
    iff_mode = np.where(sig_n >= 0, "IFF-A", np.where(np.abs(np.degrees(th_fp)) < 0.5, "IFF-B", "IFF-C"))
    mode = np.where(ff >= iff, np.where(s1 >= 0, "FFT", "FFC"), iff_mode)
    return np.maximum(ff, iff), mode, np.degrees(th_fp)


def ply_exposures(stress, strength: StrengthProperties3D, tsai_wu_interaction: Union[str, float] = "hoffman",
                  puck: PuckParameters = PuckParameters()) -> Dict[str, np.ndarray]:
    """Every criterion's exposure and mode for an array of stress states.

    Returns ``{"max_stress", "max_stress_mode", "tsai_wu", "hashin", "hashin_mode", "puck",
    "puck_mode", "puck_angle", "exposure", "margin"}``, where ``exposure`` is the largest of
    the four and ``margin = 1 / exposure - 1``.
    """
    out: Dict[str, np.ndarray] = {}
    out["max_stress"], out["max_stress_mode"] = max_stress_exposure(stress, strength)
    out["tsai_wu"] = tsai_wu_exposure(stress, strength, tsai_wu_interaction)
    out["hashin"], out["hashin_mode"] = hashin_exposure(stress, strength)
    out["puck"], out["puck_mode"], out["puck_angle"] = puck_exposure(stress, strength, puck)
    f = np.max(np.stack([out[c] for c in EXPOSURE_CRITERIA]), axis=0)
    out["exposure"] = f
    with np.errstate(divide="ignore"):
        out["margin"] = np.where(f > 0, 1.0 / np.where(f > 0, f, 1.0) - 1.0, np.inf)
    return out
