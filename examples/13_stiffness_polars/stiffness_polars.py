"""Directional stiffness of laminates: polar plots of Ex, Gxy and nu_xy.

Rotating the load direction by phi relative to a laminate is the same as rotating
every ply by -phi. For each direction the in-plane engineering constants come
from the inverse of the A matrix: Ex = 1 / (h a11), Gxy = 1 / (h a66),
nu_xy = -a12 / a11, with a = A^-1.

Four stacks of T300/5208 are compared. A quasi-isotropic stack has the same
in-plane stiffness in every direction, a cross-ply is stiff along the fibres and
soft at 45 degrees, and a +-45 stack is the opposite: soft in tension but stiff
in shear, with a Poisson's ratio well above 0.5.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import Laminate, create_default_library, integrate_abd

PLY_T = 0.125e-3
STACKS = {
    "[0]8": [0] * 8,
    "[0/90]2s": [0, 90, 0, 90, 90, 0, 90, 0],
    "[0/45/-45/90]s": [0, 45, -45, 90, 90, -45, 45, 0],
    "[±45]2s": [45, -45, 45, -45, -45, 45, -45, 45],
}


def constants(card, angles, phi):
    lam = Laminate.from_angles(card, [a - phi for a in angles], PLY_T)
    a = np.linalg.inv(integrate_abd(lam)[:3, :3])
    h = lam.thickness
    return 1 / (h * a[0, 0]), 1 / (h * a[2, 2]), -a[0, 1] / a[0, 0]


def main():
    card = create_default_library().get("T300/5208")
    phi = np.radians(np.arange(0, 361, 2.5))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), subplot_kw={"projection": "polar"})
    titles = ("Ex [GPa]", "Gxy [GPa]", "νxy")
    print(f"{'stack':<16}{'Ex 0':>8}{'Ex 45':>8}{'Gxy 0':>8}{'nu 0':>7}   [GPa]")
    for colour, (name, angles) in zip(SERIES, STACKS.items()):
        values = np.array([constants(card, angles, np.degrees(p)) for p in phi])
        ex, gxy, nu = values[:, 0] / 1e9, values[:, 1] / 1e9, values[:, 2]
        i45 = int(np.argmin(np.abs(np.degrees(phi) - 45)))
        print(f"{name:<16}{ex[0]:8.1f}{ex[i45]:8.1f}{gxy[0]:8.1f}{nu[0]:7.2f}")
        style = dict(color=colour, label=name)
        if name == "[0/45/-45/90]s":
            style.update(ls=(0, (5, 2)))
        for ax, v in zip(axes, (ex, gxy, nu)):
            ax.plot(phi, v, **style)
    for ax, title in zip(axes, titles):
        ax.set_title(title, pad=14, loc="center")
        ax.set_theta_zero_location("E")
        ax.tick_params(labelsize=7.5)
        ax.set_rlabel_position(100)
    axes[2].set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("A quasi-isotropic stack is a circle; cross-ply and ±45 trade tension for shear stiffness",
                 x=0.02, y=1.16, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.0, -0.04, "T300/5208, 8 plies of 0.125 mm. Angle = load direction measured from the 0° fibres. "
             "In-plane constants from A⁻¹.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "stiffness_polars.png")


if __name__ == "__main__":
    main()
