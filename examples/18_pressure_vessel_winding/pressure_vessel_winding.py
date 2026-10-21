"""Filament-wound pressure vessel: winding angle and material orientation on a cylinder.

A closed-end cylinder under internal pressure carries a hoop resultant pR and an
axial resultant pR/2. Netting theory says a [±θ] winding carries this best at
tan²θ = 2, θ = 54.7°. The barrel is meshed as a full 360° cylinder of Quad4
shells; ``CylindricalDirection`` sets every element's 0° direction along the
cylinder axis, so the ply angles mean the same thing all the way round.

Left: the mesh with each element's 0° direction and the outer-ply fibres. Middle:
hoop and axial mid-wall strain against winding angle, FE against classical
lamination theory (CLT). Right: first-ply failure pressure from Tsai-Wu.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import (
    DOF,
    BoundaryConditions,
    CylindricalDirection,
    CylindricalPanelBuilder,
    ExactStressRecoverer,
    FSDTModel,
    Laminate,
    LoadCase,
    StructuralGlobalSolver,
    TsaiWuCriterion,
    create_default_library,
    integrate_abd,
)

R, L, P, PLY_T = 0.15, 0.6, 1.0e6, 0.25e-3
ANGLES = [15, 25, 35, 45, 50, 54.7, 60, 65, 75, 85]


def laminate(card, theta):
    return Laminate.from_angles(card, [theta, -theta, -theta, theta], PLY_T)


def solve(card, theta):
    lam = laminate(card, theta)
    named = CylindricalPanelBuilder(R, L, 8, 32, 0.0, 360.0).build_named(
        lam, FSDTModel(), material_axis=CylindricalDirection(reference="axial")
    )
    mesh = named.mesh
    bcs = BoundaryConditions().fix(named.nodes("x0"), [DOF.UX])
    bcs.fix([mesh.closest_node((0.0, 0.0, R)), mesh.closest_node((0.0, 0.0, -R))], [DOF.UY])
    bcs.fix([mesh.closest_node((0.0, R, 0.0))], [DOF.UZ])
    loads = LoadCase().add_pressure(P)  # +normal is outward: internal pressure
    loads.add_line_load(named.nodes("x1"), (P * R / 2, 0.0, 0.0))  # end-cap thrust
    return lam, named, StructuralGlobalSolver().solve(mesh, bcs, loads)


def clt(lam, criterion):
    N = np.array([P * R / 2, P * R, 0.0])  # axial (x = 0° direction), hoop, shear
    eps = np.linalg.solve(integrate_abd(lam)[:3, :3], N)
    ratio = min(
        criterion.evaluate(ply.to_material_stress(ply.Q_bar @ eps), ply.material.strength).strength_ratio
        for ply in lam.plies
    )
    return eps, ratio


def main():
    card = create_default_library().get("T300/5208")
    criterion = TsaiWuCriterion()
    recoverer = ExactStressRecoverer()
    fe, theory = [], []
    keep = None
    print(f"{'theta':>6}{'eps_axial FE':>14}{'CLT':>11}{'eps_hoop FE':>13}{'CLT':>11}{'burst FE [MPa]':>16}{'CLT':>8}")
    for theta in ANGLES:
        lam, named, result = solve(card, theta)
        mesh = named.mesh
        middle = [e for e in mesh.elements if abs(e.centroid[0] - L / 2) < L / 8]
        strains = np.mean([e.generalized_strains(result.element_displacements(e.element_id))[0] for e in middle], axis=0)
        ratio = min(
            criterion.evaluate(p.stress_material, lam.plies[p.ply_index].material.strength).strength_ratio
            for e in middle
            for p in recoverer.ply_midplane_states(e, result.element_displacements(e.element_id))
        )
        eps_clt, ratio_clt = clt(lam, criterion)
        fe.append((strains[0], strains[1], P * ratio))
        theory.append((eps_clt[0], eps_clt[1], P * ratio_clt))
        print(f"{theta:6.1f}{strains[0]:14.3e}{eps_clt[0]:11.3e}{strains[1]:13.3e}{eps_clt[1]:11.3e}"
              f"{P * ratio / 1e6:16.2f}{P * ratio_clt / 1e6:8.2f}")
        if theta == 54.7:
            keep = (named, result)
    fe, theory = np.array(fe), np.array(theory)

    fig = plt.figure(figsize=(13.0, 4.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.1])
    ax0, ax1, ax2 = (fig.add_subplot(grid[0, i]) for i in range(3))

    named, result = keep
    mesh = named.mesh
    ax0.remove()
    ax0 = fig.add_subplot(grid[0, 0], projection="3d")
    for e in mesh.elements:
        loop = np.vstack([e.coordinates, e.coordinates[:1]]) * 1e3
        ax0.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#c9c8c4", lw=0.4)
    # outer-ply fibre direction: the element 0-degree axis turned by +theta about the normal
    theta = np.radians(54.7)
    for e in mesh.elements[::3]:
        e1, e2, _ = e.frame
        fibre = np.cos(theta) * e1 + np.sin(theta) * e2
        c = e.centroid * 1e3
        ax0.quiver(*c, *(fibre * 45), color=SERIES[0], lw=1.0, arrow_length_ratio=0.0)
        ax0.quiver(*c, *(e1 * 45), color=SERIES[1], lw=0.8, arrow_length_ratio=0.0)
    ax0.plot([], [], color=SERIES[1], label="0° direction (cylinder axis)")
    ax0.plot([], [], color=SERIES[0], label="+54.7° fibres")
    ax0.set_box_aspect((L, 2 * R, 2 * R))
    ax0.set_axis_off()
    ax0.view_init(elev=22, azim=-60)
    ax0.set_title("CylindricalDirection sets the 0° axis", loc="left")
    ax0.legend(loc="lower left", fontsize=8)

    ax1.plot(ANGLES, fe[:, 1] * 1e3, "o", color=SERIES[0], label="hoop strain, FE")
    ax1.plot(ANGLES, theory[:, 1] * 1e3, color=SERIES[0], lw=1.2, ls=(0, (4, 2)), label="hoop strain, CLT")
    ax1.plot(ANGLES, fe[:, 0] * 1e3, "o", color=SERIES[1], label="axial strain, FE")
    ax1.plot(ANGLES, theory[:, 0] * 1e3, color=SERIES[1], lw=1.2, ls=(0, (4, 2)), label="axial strain, CLT")
    ax1.axhline(0.0, color=TEXT_SECONDARY, lw=0.8)
    ax1.axvline(54.7, color=TEXT_SECONDARY, lw=0.8, ls=(0, (2, 2)))
    ax1.set_xlabel("winding angle θ from the axis [°]")
    ax1.set_ylabel("mid-wall strain [‰]")
    ax1.set_title("FE matches CLT; axial strain flips near 52°", loc="left")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(ANGLES, fe[:, 2] / 1e6, "o-", color=SERIES[2], label="FE + Tsai-Wu")
    ax2.plot(ANGLES, theory[:, 2] / 1e6, color=REFERENCE, lw=1.2, ls=(0, (4, 2)), label="CLT + Tsai-Wu")
    best = ANGLES[int(np.argmax(fe[:, 2]))]
    ax2.axvline(54.7, color=TEXT_SECONDARY, lw=0.8, ls=(0, (2, 2)))
    ax2.text(56.0, 0.92 * fe[:, 2].max() / 1e6, "netting\ntheory\n54.7°", fontsize=8, color=TEXT_SECONDARY)
    ax2.set_xlabel("winding angle θ from the axis [°]")
    ax2.set_ylabel("first-ply failure pressure [MPa]")
    ax2.set_title(f"Strongest winding: ±{best:g}°", loc="left")
    ax2.legend(loc="upper left", fontsize=8)
    fig.text(0.0, -0.04, f"[±θ]s T300/5208, {PLY_T * 1e3:.2f} mm plies, R = {R * 1e3:.0f} mm, L = {L * 1e3:.0f} mm, "
             f"p = {P / 1e6:.0f} MPa with end-cap thrust pR/2, 8 x 32 Quad4 around the full barrel, FSDT.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "pressure_vessel_winding.png")


if __name__ == "__main__":
    main()
