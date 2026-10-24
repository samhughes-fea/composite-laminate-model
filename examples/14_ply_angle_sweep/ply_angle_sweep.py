"""Ply-angle sweep: which [+theta/-theta]2s laminate carries a given load best?

For theta from 0 to 90 degrees, the first-ply failure load (Tsai-Wu) of an
eight-ply [+theta/-theta]2s T300/5208 laminate is found under three load
cases. The best angle depends entirely on the load: near 0 degrees for
uniaxial tension, 45 degrees for shear, and a compromise for combined loads.
(Tsai-Wu puts the uniaxial optimum at 10 rather than 0 degrees because its
quadratic surface lets fibres exceed Xt when the matrix is in transverse
compression; see example 05.) This is
the simplest form of laminate design optimisation, and every point is a
linear static analysis.

Each laminate is analysed on a single free Quad4 element (``_shared/membrane.py``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.membrane import MembraneCell
from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import Laminate, TsaiWuCriterion, create_default_library

PLY_T = 0.125e-3
ANGLES = np.arange(0, 91, 5)
LOAD_CASES = {
    "N_x": np.array([1.0, 0.0, 0.0]),
    "N_x = N_y": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
    "N_xy (shear)": np.array([0.0, 0.0, 1.0]),
    "N_x : N_xy = 2 : 1": np.array([2.0, 0.0, 1.0]) / np.sqrt(5),
}


def main():
    card = create_default_library().get("T300/5208")
    criterion = TsaiWuCriterion()
    results = {name: [] for name in LOAD_CASES}
    for theta in ANGLES:
        lam = Laminate.from_angles(card, [theta, -theta] * 2 + [-theta, theta] * 2, PLY_T)
        cell = MembraneCell.build(lam)
        for name, direction in LOAD_CASES.items():
            results[name].append(cell.failure_load(criterion, direction, include_residual=False)[0] / 1e3)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    print(f"{'load case':<20}{'best theta':>11}{'FPF [kN/m]':>12}")
    for colour, (name, values) in zip(SERIES, results.items()):
        values = np.array(values)
        best = int(np.argmax(values))
        print(f"{name:<20}{ANGLES[best]:11.0f}{values[best]:12.0f}")
        ax.plot(ANGLES, values, marker="o", ms=4, color=colour, label=f"{name}: best θ = {ANGLES[best]}°")
        ax.plot(ANGLES[best], values[best], "o", ms=11, mfc="none", mec=colour, mew=1.6)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_xlabel("ply angle θ in [+θ/-θ]2s [deg]")
    ax.set_ylabel("first-ply failure load |N| [kN/m]")
    ax.set_title("The best ply angle depends on the load")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.text(0.0, -0.03, f"T300/5208, 8 plies of {PLY_T * 1e3:.3f} mm, Tsai-Wu (Hoffman F12), no residual stress. "
             "|N| is the magnitude of the load vector (N_x, N_y, N_xy).", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "ply_angle_sweep.png")


if __name__ == "__main__":
    main()
