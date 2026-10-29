"""Thermal warping of unsymmetric laminates after a cure cool-down.

Three free 100 mm square T300/5208 plates are cooled by 150 K. The stacking
sequence alone decides the shape:

* [0/90] has B11 and B22 coupling, so it curls into a saddle;
* [45/-45] has B16 and B26 coupling, so it twists;
* [0/90]s is symmetric, so B = 0 and it only shrinks in plane.

The centre node is fully restrained, which removes rigid-body motion without
adding any stress. The finite-element curvatures are compared with the
laminate-theory result {eps0; kappa} = ABD^-1 {N_ht; M_ht}.

This is linear theory. Real thin [0/90] plates of this size snap into a
cylinder (a geometrically nonlinear effect), so read the saddle as the linear
prediction, not the final cured shape.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import DIVERGING, TEXT_SECONDARY, plt, save
from composite_laminate_model import (
    DOF,
    BoundaryConditions,
    FEAMesh,
    FSDTModel,
    HygrothermalEngine,
    HygrothermalState,
    Laminate,
    LoadCase,
    ResultsExtractor,
    StructuralGlobalSolver,
    create_default_library,
    integrate_abd,
)

A, N_ELEM, PLY_T, DELTA_T = 0.1, 16, 0.125e-3, -150.0
STACKS = {"[0/90]": [0, 90], "[45/-45]": [45, -45], "[0/90]s": [0, 90, 90, 0]}


def solve(laminate):
    mesh = FEAMesh.rectangular_plate(A, A, N_ELEM, N_ELEM, laminate, FSDTModel())
    centre = mesh.closest_node((A / 2, A / 2))
    bcs = BoundaryConditions().fix([centre], [DOF.UX, DOF.UY, DOF.UZ, DOF.RX, DOF.RY, DOF.RZ])
    loads = LoadCase().set_hygrothermal(HygrothermalState(delta_temperature=DELTA_T))
    return mesh, ResultsExtractor(StructuralGlobalSolver().solve(mesh, bcs, loads), loads)


def main():
    card = create_default_library().get("T300/5208")
    state = HygrothermalState(delta_temperature=DELTA_T)
    fig = plt.figure(figsize=(13.0, 5.0))
    w_max = 0.0
    results = {}
    print(f"{'stack':<10}{'kx FE':>11}{'kx CLT':>11}{'ky FE':>11}{'ky CLT':>11}{'kxy FE':>11}{'kxy CLT':>11}   [1/m]")
    for name, angles in STACKS.items():
        lam = Laminate.from_angles(card, angles, PLY_T)
        mesh, ex = solve(lam)
        nodes, gp = ex.nodes(), ex.gauss_points()
        theory = np.linalg.solve(integrate_abd(lam), HygrothermalEngine().resultant_vector(lam, state))
        fe = gp.kappa.mean(axis=0)
        print(f"{name:<10}" + "".join(f"{fe[i]:11.4f}{theory[3 + i]:11.4f}" for i in range(3)))
        results[name] = (nodes, fe, theory)
        w_max = max(w_max, np.abs(nodes.displacement[:, 2]).max())

    for i, (name, (nodes, fe, theory)) in enumerate(results.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        x, y = nodes.xyz[:, 0] * 1e3, nodes.xyz[:, 1] * 1e3
        w = nodes.displacement[:, 2] * 1e3
        surface = ax.plot_trisurf(x, y, w, cmap=DIVERGING, vmin=-w_max * 1e3, vmax=w_max * 1e3, linewidth=0.1,
                                  edgecolor="#fcfcfb", antialiased=True)
        ax.set_zlim(-w_max * 1e3, w_max * 1e3)
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("y [mm]", fontsize=8)
        ax.set_zlabel("w [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=24, azim=-58)
        ax.set_facecolor("#fcfcfb")
        parts = [f"{label} = {value:+.1f}" for label, value in zip(("κx", "κy", "κxy"), fe) if abs(value) > 1e-6]
        subtitle = ", ".join(parts) + " 1/m" if parts else "flat, B = 0"
        ax.set_title(f"{name}\n{subtitle}", fontsize=10, loc="center")
    fig.subplots_adjust(left=0.0, right=0.9, top=0.8, wspace=0.08)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
    fig.colorbar(surface, cax=cax, label="w [mm]")
    fig.suptitle(f"A {abs(DELTA_T):.0f} K cool-down warps unsymmetric laminates: saddle, twist, or flat",
                 x=0.02, y=0.98, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.02, 0.0, f"T300/5208, {A * 1e3:.0f} mm square, ply {PLY_T * 1e3:.3f} mm, free plate (centre node held), "
             f"{N_ELEM} x {N_ELEM} FSDT mesh. Linear theory; displacements are true scale.", fontsize=7.5, color=TEXT_SECONDARY)
    save(fig, __file__, "thermal_warping.png")


if __name__ == "__main__":
    main()
