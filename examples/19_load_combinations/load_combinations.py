"""Load cases and combinations on a cantilevered composite panel with a strut.

A 1 m x 0.25 m panel is clamped at the root and propped by a strut (a grounded
spring) at mid-span. Four load cases use the load types the solver supports:

* ``dead``:  self-weight as a body load (g along -z);
* ``lift``:  an elliptic pressure distribution along the span (a callable pressure);
* ``tip``:   a line load on the tip edge;
* ``cold``:  a uniform temperature drop of 50 K.

``solve_many`` assembles and factorises the stiffness once for all cases.
``solve_combinations`` then forms design combinations by superposition, which is
exact for a linear analysis.

Left: tip deflection for each case and combination. Middle: deflection along the
span. Right: the highest Tsai-Wu index in the root plies for each combination.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import (
    DOF,
    BoundaryConditions,
    FSDTModel,
    HygrothermalState,
    Laminate,
    LoadCase,
    LoadCombination,
    RectangularPlateBuilder,
    ExactStressRecoverer,
    StructuralGlobalSolver,
    TsaiWuCriterion,
    create_default_library,
)

SPAN, CHORD, NX, NY = 1.0, 0.25, 40, 10
P0, TIP, STRUT_K = 1.0e3, 50.0, 5.0e5


def build(card):
    lam = Laminate.from_angles(card, [0, 45, -45, 90, 0] * 2 + [0, 90, -45, 45, 0] * 2, 0.25e-3)
    named = RectangularPlateBuilder(SPAN, CHORD, NX, NY).build_named(lam, FSDTModel())
    bcs = BoundaryConditions().clamp(named.nodes("left"))
    strut = named.mesh.closest_node((SPAN / 2, CHORD / 2))
    bcs.add_spring([strut], [DOF.UZ], STRUT_K)
    return lam, named, bcs


def cases(named):
    tip_nodes = named.nodes("right")
    return [
        LoadCase("dead").add_body_load((0.0, 0.0, -9.81)),
        LoadCase("lift").add_pressure(lambda x: P0 * np.sqrt(max(0.0, 1.0 - (x[0] / SPAN) ** 2))),
        LoadCase("tip").add_line_load(tip_nodes, (0.0, 0.0, -TIP / CHORD)),
        LoadCase("cold").set_hygrothermal(HygrothermalState(-50.0)),
    ]


COMBINATIONS = [
    LoadCombination("ULS up", {"dead": 1.0, "lift": 1.5, "cold": 1.0}),
    LoadCombination("ULS down", {"dead": 1.35, "tip": 1.5, "cold": 1.0}),
    LoadCombination("SLS", {"dead": 1.0, "lift": 1.0, "tip": 1.0}),
]


def main():
    card = create_default_library().get("T300/5208")
    lam, named, bcs = build(card)
    mesh = named.mesh
    solver = StructuralGlobalSolver()
    load_cases = cases(named)
    results = dict(zip([c.name for c in load_cases], solver.solve_many(mesh, bcs, load_cases)))
    results.update(solver.solve_combinations(mesh, bcs, load_cases, COMBINATIONS))

    # superposition check against a direct solve of one combination
    direct = LoadCase("direct").add_body_load((0.0, 0.0, -9.81)).add_pressure(
        lambda x: 1.5 * P0 * np.sqrt(max(0.0, 1.0 - (x[0] / SPAN) ** 2))).set_hygrothermal(HygrothermalState(-50.0))
    reference = solver.solve(mesh, bcs, direct).displacements
    diff = np.abs(results["ULS up"].displacements - reference).max() / np.abs(reference).max()
    print(f"'ULS up' by superposition vs a direct solve: max relative difference {diff:.1e}")

    centre = sorted(mesh.nodes_on_line(1, CHORD / 2), key=lambda n: mesh.node_coordinates(n)[0])
    x = np.array([mesh.node_coordinates(n)[0] for n in centre])
    tip = mesh.closest_node((SPAN, CHORD / 2))
    print(f"{'case':<10}{'tip w [mm]':>12}{'strut force [N]':>17}")
    strut = mesh.closest_node((SPAN / 2, CHORD / 2))
    for name, r in results.items():
        print(f"{name:<10}{r.node_displacement(tip)[2] * 1e3:12.2f}{-STRUT_K * r.node_displacement(strut)[2]:17.1f}")

    # governing combination: highest Tsai-Wu index in any ply of the root elements
    recoverer, criterion = ExactStressRecoverer(), TsaiWuCriterion()
    root = [e for e in mesh.elements if e.centroid[0] < SPAN / NX]
    governing = {}
    for combo in COMBINATIONS:
        r = results[combo.name]
        state = HygrothermalState(-50.0 * combo.factors.get("cold", 0.0))
        governing[combo.name] = max(
            criterion.evaluate(p.stress_material, lam.plies[p.ply_index].material.strength).index
            for e in root
            for p in recoverer.ply_midplane_states(e, r.element_displacements(e.element_id), state=state)
        )
        print(f"{combo.name:<10} max Tsai-Wu index at the root {governing[combo.name]:.3f}")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.0, 4.3), gridspec_kw={"width_ratios": [1.1, 1.2, 1.0]})
    names = list(results)
    tips = [results[n].node_displacement(tip)[2] * 1e3 for n in names]
    colours = [TEXT_SECONDARY] * len(load_cases) + SERIES[: len(COMBINATIONS)]
    bars = ax0.barh(range(len(names)), tips, color=colours, height=0.6)
    for bar, value in zip(bars, tips):
        ax0.text(value + (0.3 if value >= 0 else -0.3), bar.get_y() + bar.get_height() / 2, f"{value:.1f}",
                 va="center", ha="left" if value >= 0 else "right", fontsize=8, color=TEXT_SECONDARY)
    ax0.set_yticks(range(len(names)))
    ax0.set_yticklabels(names)
    ax0.invert_yaxis()
    ax0.axvline(0.0, color=TEXT_SECONDARY, lw=0.8)
    ax0.set_xlabel("tip deflection w [mm]")
    ax0.set_title("Cases (grey) and combinations", loc="left")
    ax0.grid(axis="y", visible=False)
    lo, hi = min(tips), max(tips)
    ax0.set_xlim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))

    for colour, combo in zip(SERIES, COMBINATIONS):
        w = [results[combo.name].node_displacement(n)[2] * 1e3 for n in centre]
        ax1.plot(x * 1e3, w, color=colour, label=combo.name)
    ax1.axvline(SPAN / 2 * 1e3, color=TEXT_SECONDARY, lw=0.8, ls=(0, (2, 2)))
    ax1.text(SPAN / 2 * 1e3 + 10, ax1.get_ylim()[0] * 0.9, "strut", fontsize=8, color=TEXT_SECONDARY)
    ax1.axhline(0.0, color=TEXT_SECONDARY, lw=0.8)
    ax1.set_xlabel("span position x [mm]")
    ax1.set_ylabel("w on the centre line [mm]")
    ax1.set_title("Deflection of each combination", loc="left")
    ax1.legend(loc="upper left")

    order = list(governing)
    values = [governing[n] for n in order]
    ax2.bar(range(len(order)), values, color=SERIES[: len(order)], width=0.6)
    for k, v in enumerate(values):
        ax2.text(k, v * 1.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order)
    ax2.set_ylabel("max Tsai-Wu index, root plies")
    ax2.set_title(f"'{order[int(np.argmax(values))]}' governs", loc="left")
    ax2.grid(axis="x", visible=False)
    fig.text(0.0, -0.04, f"[0/45/-45/90/0]2s T300/5208 (5 mm), {SPAN * 1e3:.0f} x {CHORD * 1e3:.0f} mm, clamped root, "
             f"strut k = {STRUT_K / 1e3:.0f} kN/m at mid-span, {NX} x {NY} Quad4, FSDT. Lift p = {P0 / 1e3:.0f} kPa "
             f"x sqrt(1 - (x/L)²), tip load {TIP:.0f} N, cold = -50 K.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "load_combinations.png")


if __name__ == "__main__":
    main()
