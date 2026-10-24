"""First-ply failure envelopes of whole laminates in the N_x-N_y plane, with and without cure stress.

Each laminate is loaded along 144 radial paths in (N_x, N_y), and the line load
at which the first ply reaches Tsai-Wu failure is found by bisection. The
analysis is repeated with the residual stress from a 150 K cure cool-down
locked in. That residual stress puts the matrix of every off-axis ply into
transverse tension before any load is applied, and it can halve the first-ply
failure load.

The laminate is a single free Quad4 element (``_shared/membrane.py``): ply
stresses for unit N_x, N_y and for the cool-down are recovered once, and any
combined load is a linear combination of them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.membrane import MembraneCell
from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import HygrothermalState, Laminate, TsaiWuCriterion, create_default_library

PLY_T, DELTA_T = 0.125e-3, -150.0
LAMINATES = {
    "[0/90]2s": [0, 90, 0, 90, 90, 0, 90, 0],
    "[0/45/-45/90]s": [0, 45, -45, 90, 90, -45, 45, 0],
    "[±45]2s": [45, -45, 45, -45, -45, 45, -45, 45],
}


def envelope(cell, criterion, residual, n=144):
    theta = np.linspace(0, 2 * np.pi, n + 1)
    points, plies = [], []
    for t in theta:
        direction = np.array([np.cos(t), np.sin(t), 0.0])
        scale, ply = cell.failure_load(criterion, direction, include_residual=residual)
        points.append(scale * direction[:2])
        plies.append(ply)
    return np.array(points) / 1e3, np.array(plies)  # kN/m


def main():
    card = create_default_library().get("T300/5208")
    criterion = TsaiWuCriterion()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
    print(f"{'laminate':<16}{'Nx FPF clean':>14}{'Nx FPF cured':>14}   [kN/m]")
    for ax, (name, angles) in zip(axes, LAMINATES.items()):
        lam = Laminate.from_angles(card, angles, PLY_T)
        cell = MembraneCell.build(lam, HygrothermalState(delta_temperature=DELTA_T))
        clean, _ = envelope(cell, criterion, residual=False)
        cured, plies = envelope(cell, criterion, residual=True)
        nx_clean = cell.failure_load(criterion, [1, 0, 0], include_residual=False)[0] / 1e3
        nx_cured = cell.failure_load(criterion, [1, 0, 0], include_residual=True)[0] / 1e3
        print(f"{name:<16}{nx_clean:14.0f}{nx_cured:14.0f}")
        ax.plot(clean[:, 0], clean[:, 1], color=SERIES[0], label="no residual stress")
        ax.plot(cured[:, 0], cured[:, 1], color=SERIES[1], label=f"after {abs(DELTA_T):.0f} K cool-down")
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("N_x [kN/m]")
        drop = 100 * (1 - nx_cured / nx_clean)
        ax.set_title(f"{name}: cure stress cuts N_x FPF {drop:.0f}%")
    axes[0].set_ylabel("N_y [kN/m]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.text(0.0, -0.03, f"T300/5208, 8 plies of {PLY_T * 1e3:.3f} mm, Tsai-Wu (Hoffman F12), N_xy = 0. Cure residual stress "
             f"from a uniform ΔT = {DELTA_T:.0f} K, linear elastic.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "laminate_failure_envelope.png")


if __name__ == "__main__":
    main()
