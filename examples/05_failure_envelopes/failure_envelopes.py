"""Failure envelopes of a unidirectional T300/5208 ply under plane stress.

Each criterion is evaluated along 720 radial load paths in two stress planes.
The ``strength_ratio`` returned by every criterion is the factor by which the
applied stress can be scaled before failure, so ``strength_ratio x direction``
is a point on the envelope.

* Left, sigma1-sigma2 (tau12 = 0): fibre direction versus transverse direction.
* Right, sigma2-tau12 (sigma1 = 0): the matrix-dominated plane, where the
  criteria disagree most because transverse compression raises shear strength
  in some criteria and not in others.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save
from composite_laminate_model import (
    HashinCriterion,
    MaximumStressCriterion,
    TsaiHillCriterion,
    TsaiWuCriterion,
    create_default_library,
)

CRITERIA = {
    "Maximum stress": MaximumStressCriterion(),
    "Tsai-Hill": TsaiHillCriterion(),
    "Tsai-Wu": TsaiWuCriterion(),
    "Hashin 2D": HashinCriterion("2d"),
}
STYLES = {
    "Maximum stress": dict(color=SERIES[0]),
    "Tsai-Hill": dict(color=SERIES[1]),
    "Tsai-Wu": dict(color=SERIES[2]),
    "Hashin 2D": dict(color=SERIES[3], ls=(0, (5, 2))),
}
# Voigt order in material axes: [s11, s22, s33, t23, t13, t12]
PLANES = {"s1-s2": (0, 1), "s2-t12": (1, 5)}


def envelope(criterion, strength, components, n=720):
    theta = np.linspace(0.0, 2.0 * np.pi, n + 1)
    points = []
    for t in theta:
        stress = np.zeros(6)
        stress[components[0]], stress[components[1]] = np.cos(t), np.sin(t)
        ratio = criterion.evaluate(stress * 1e6, strength).strength_ratio * 1e6
        points.append(ratio * np.array([np.cos(t), np.sin(t)]))
    return np.array(points) / 1e6  # MPa


def main():
    strength = create_default_library().get("T300/5208").strength
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
    for name, criterion in CRITERIA.items():
        e1 = envelope(criterion, strength, PLANES["s1-s2"])
        e2 = envelope(criterion, strength, PLANES["s2-t12"])
        ax1.plot(e1[:, 0], e1[:, 1], label=name, **STYLES[name])
        ax2.plot(e2[:, 0], e2[:, 1], label=name, **STYLES[name])
        at = e2[np.argmin(np.abs(np.arctan2(e2[:, 1], e2[:, 0]) - np.arctan2(1.0, -1.0)))]
        print(f"{name:<15} sigma2 = tau12 compressive-shear failure at |s| = {np.hypot(*at):6.1f} MPa")

    mpa = lambda v: v / 1e6
    ax1.set_xlabel("σ1, fibre direction [MPa]")
    ax1.set_ylabel("σ2, transverse [MPa]")
    ax1.set_title("σ1–σ2: Hashin and maximum stress coincide")
    ax1.set_xlim(-1.15 * mpa(strength.Xc), 1.15 * mpa(strength.Xt))
    ax1.set_ylim(-1.25 * mpa(strength.Yc), 2.5 * mpa(strength.Yt))
    ax2.set_xlabel("σ2, transverse [MPa]")
    ax2.set_ylabel("τ12, in-plane shear [MPa]")
    ax2.set_title("σ2–τ12: Tsai-Wu gains shear strength in compression")
    ax2.set_xlim(-1.15 * mpa(strength.Yc), 1.6 * mpa(strength.Yt))
    ax2.set_ylim(-1.3 * mpa(strength.S12), 1.3 * mpa(strength.S12))
    for ax in (ax1, ax2):
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.text(0.0, -0.03, f"T300/5208: Xt = {mpa(strength.Xt):.0f}, Xc = {mpa(strength.Xc):.0f}, Yt = {mpa(strength.Yt):.0f}, "
             f"Yc = {mpa(strength.Yc):.0f}, S12 = {mpa(strength.S12):.0f} MPa. Tsai-Wu uses the Hoffman F12 term.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "failure_envelopes.png")


if __name__ == "__main__":
    main()
