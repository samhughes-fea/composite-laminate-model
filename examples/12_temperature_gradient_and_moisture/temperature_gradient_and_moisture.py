"""Through-thickness temperature gradients and moisture swelling.

Left: a symmetric [0/90]s plate has no thermal warping under a uniform
temperature change (example 04), but a temperature difference between its
faces still bends it. Here the top face is 40 K hotter and the bottom face
40 K colder than the stress-free temperature. The plate curls almost into a
cylinder: the outer 0-degree plies barely expand along the fibres, so nearly all
the curvature is about the x axis (in kappa_y).

Right: an unsymmetric [0/90] plate warps on cure cool-down. The matrix then
absorbs moisture in service and swells, which works against the thermal
shrinkage. Laminate theory predicts the moisture content at which the
curvature returns to zero; the FE points follow it exactly.

Both cases use ``HygrothermalState(delta_temperature, delta_moisture,
temperature_gradient, moisture_gradient)``. The fields at height z are
dT(z) = delta_temperature + temperature_gradient * z (and likewise for moisture).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import DIVERGING, REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import (
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

A, N_ELEM, PLY_T = 0.1, 12, 0.25e-3
FACE_DT, CURE_DT = 40.0, -150.0
MOISTURE = np.linspace(0.0, 0.012, 9)  # mass fraction of absorbed moisture


def free_plate(lam, state):
    mesh = FEAMesh.rectangular_plate(A, A, N_ELEM, N_ELEM, lam, FSDTModel())
    centre = mesh.closest_node((A / 2, A / 2))
    bcs = BoundaryConditions().fix([centre], [DOF.UX, DOF.UY, DOF.UZ, DOF.RX, DOF.RY, DOF.RZ])
    loads = LoadCase().set_hygrothermal(state)
    return mesh, ResultsExtractor(StructuralGlobalSolver().solve(mesh, bcs, loads), loads)


def main():
    card = create_default_library().get("T300/5208")
    fig = plt.figure(figsize=(12.0, 4.8))

    sym = Laminate.from_angles(card, [0, 90, 90, 0], PLY_T)
    gradient = 2 * FACE_DT / sym.thickness  # K/m: +FACE_DT on top, -FACE_DT at the bottom
    state = HygrothermalState(temperature_gradient=gradient)
    mesh, ex = free_plate(sym, state)
    nodes = ex.nodes()
    fe_kappa = ex.gauss_points().kappa.mean(axis=0)
    theory = np.linalg.solve(integrate_abd(sym), HygrothermalEngine().resultant_vector(sym, state))[3:]
    print(f"[0/90]s gradient: kappa FE {fe_kappa.round(4)}  CLT {theory.round(4)} 1/m")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    w = nodes.displacement[:, 2] * 1e3
    lim = np.abs(w).max()
    ax.plot_trisurf(nodes.xyz[:, 0] * 1e3, nodes.xyz[:, 1] * 1e3, w, cmap=DIVERGING, vmin=-lim, vmax=lim,
                    linewidth=0.1, edgecolor="#fcfcfb")
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.set_zlabel("w [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=-60)
    ax.set_title(f"[0/90]s, faces at ±{FACE_DT:.0f} K: κx = {fe_kappa[0]:.2f}, κy = {fe_kappa[1]:.2f} 1/m",
                 fontsize=10, loc="center")

    ax2 = fig.add_subplot(1, 2, 2)
    unsym = Laminate.from_angles(card, [0, 90], PLY_T)
    abd = integrate_abd(unsym)
    fe, clt = [], []
    for c in MOISTURE:
        s = HygrothermalState(delta_temperature=CURE_DT, delta_moisture=c)
        fe.append(free_plate(unsym, s)[1].gauss_points().kappa.mean(axis=0)[0])
    fine = np.linspace(MOISTURE[0], MOISTURE[-1], 100)
    for c in fine:
        s = HygrothermalState(delta_temperature=CURE_DT, delta_moisture=c)
        clt.append(np.linalg.solve(abd, HygrothermalEngine().resultant_vector(unsym, s))[3])
    clt = np.array(clt)
    zero = np.interp(0.0, clt if clt[0] < clt[-1] else clt[::-1], fine if clt[0] < clt[-1] else fine[::-1])
    print(f"[0/90] curvature vanishes at {zero * 100:.2f}% moisture")
    ax2.plot(fine * 100, clt, color=REFERENCE, lw=1.2, ls=(0, (4, 3)), label="laminate theory")
    ax2.plot(MOISTURE * 100, fe, "o", color=SERIES[0], label="FE")
    ax2.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    ax2.axvline(zero * 100, color=TEXT_SECONDARY, lw=0.8, ls=(0, (1, 2)))
    ax2.set_xlabel("absorbed moisture [% by mass]")
    ax2.set_ylabel("κx [1/m]")
    ax2.set_title(f"Moisture undoes the cure warp of [0/90] at {zero * 100:.2f}%")
    ax2.legend(loc="lower right")
    fig.suptitle("Temperature gradients bend symmetric laminates; moisture swelling offsets thermal shrinkage",
                 x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.02, -0.02, f"T300/5208 (α2 = 22.5e-6 /K, β2 = 0.6), {PLY_T * 1e3:.2f} mm plies, {A * 1e3:.0f} mm square free plates "
             f"(centre node held), {N_ELEM} x {N_ELEM} FSDT. Right: ΔT = {CURE_DT:.0f} K plus uniform moisture.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    save(fig, __file__, "temperature_gradient_and_moisture.png")


if __name__ == "__main__":
    main()
