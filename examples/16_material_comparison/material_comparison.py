"""Material comparison: one plate, five materials from the library.

A simply supported 200 mm square plate, 2 mm thick, carries a uniform pressure.
It is built from each card in the default material library: three carbon/epoxy
systems and E-glass/epoxy as an [0/45/-45/90]s quasi-isotropic stack, and
6061-T6 aluminium as a single 2 mm sheet. For each material the example reports:

* the centre deflection,
* the pressure at first-ply failure (Tsai-Wu strength ratio x applied pressure,
  from ``ResultsExtractor`` ply stresses),
* the mass per unit area.

Swapping materials is one line, because a ``MaterialCard`` carries elastic,
strength, hygrothermal and density data together.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save, simply_supported
from laminate_modelling_package import (
    FEAMesh,
    FSDTModel,
    Laminate,
    LoadCase,
    ResultsExtractor,
    StructuralGlobalSolver,
    TsaiWuCriterion,
    create_default_library,
)

A, H, Q, N_ELEM = 0.2, 2.0e-3, 5.0e3, 12
QUASI_ISO = [0, 45, -45, 90, 90, -45, 45, 0]


def analyse(card, criterion):
    angles = [0] if card.name.startswith("Al") else QUASI_ISO
    lam = Laminate.from_angles(card, angles, H / len(angles))
    mesh = FEAMesh.rectangular_plate(A, A, N_ELEM, N_ELEM, lam, FSDTModel())
    result = StructuralGlobalSolver().solve(mesh, simply_supported(mesh, A, A), LoadCase().add_pressure(Q))
    w = result.node_displacement(mesh.closest_node((A / 2, A / 2)))[2]
    plies = ResultsExtractor(result).ply_points()
    ratio = np.inf
    for k, s, t in zip(plies.ply_index, plies.stress_material, plies.shear_material):
        stress = np.array([s[0], s[1], 0.0, t[0], t[1], s[2]])
        ratio = min(ratio, criterion.evaluate(stress, lam.plies[k].material.strength).strength_ratio)
    return w, ratio * Q, card.density * H


def main():
    library = create_default_library()
    criterion = TsaiWuCriterion()
    names = library.names()
    data = {name: analyse(library.get(name), criterion) for name in names}
    print(f"{'material':<16}{'w [mm]':>9}{'p_FPF [kPa]':>13}{'mass [kg/m2]':>14}{'p_FPF / mass':>14}")
    for name, (w, p, m) in data.items():
        print(f"{name:<16}{w * 1e3:9.2f}{p / 1e3:13.0f}{m:14.2f}{p / 1e3 / m:14.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    pos = np.arange(len(names))
    panels = [
        ([data[n][0] * 1e3 for n in names], f"centre deflection at {Q / 1e3:.0f} kPa [mm]", "Deflection: aluminium ≈ quasi-isotropic carbon", "{:.2f}"),
        ([data[n][1] / 1e3 for n in names], "first-ply failure pressure [kPa]", "Strength: IM7/8552 carries the most pressure", "{:.0f}"),
        ([data[n][1] / 1e3 / data[n][2] for n in names], "failure pressure per mass [kPa per kg/m²]",
         "Per kilogram, carbon beats aluminium", "{:.0f}"),
    ]
    for ax, (values, ylabel, title, fmt) in zip(axes, panels):
        ax.bar(pos, values, color=[SERIES[0]] * 3 + [SERIES[1], SERIES[2]], width=0.6)
        for p, v in zip(pos, values):
            ax.text(p, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xticks(pos)
        ax.set_xticklabels(names, rotation=20, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    fig.text(0.0, -0.06, f"{A * 1e3:.0f} mm square, {H * 1e3:.0f} mm thick, simply supported, q = {Q / 1e3:.0f} kPa, "
             f"{N_ELEM} x {N_ELEM} FSDT. Composites [0/45/-45/90]s; aluminium one isotropic layer. Tsai-Wu for all "
             "(aluminium strength = yield). Linear small-deflection theory.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "material_comparison.png")


if __name__ == "__main__":
    main()
