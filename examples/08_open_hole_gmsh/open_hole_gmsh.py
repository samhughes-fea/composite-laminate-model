"""Open-hole tension meshed with Gmsh: stress concentration of four laminates.

A 300 x 150 mm plate with a central 10 mm hole is meshed by Gmsh into quads that
are graded towards the hole, then pulled along x. The stress concentration
factor Kt depends on the laminate's in-plane orthotropy, which Lekhnitskii's
solution for an infinite orthotropic plate captures:

    Kt_inf = 1 + sqrt(2 (sqrt(Ex/Ey) - nu_xy) + Ex/Gxy)

The finite-width correction Kt = Kt_inf * (2 + (1 - d/W)^3) / (3 (1 - d/W)) is
applied, and the stress decay away from the hole is compared with the
Konish-Whitney approximation.

Needs Gmsh: pip install "composite-laminate-model[gmsh]".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib.collections import PolyCollection

from _shared.plotting import REFERENCE, SEQUENTIAL, SERIES, TEXT_SECONDARY, plt, save
from composite_laminate_model import (
    DOF,
    BoundaryConditions,
    CircularHole,
    FSDTModel,
    GmshMeshImporter,
    GmshPlateBuilder,
    Laminate,
    LoadCase,
    ResultsExtractor,
    StructuralGlobalSolver,
    create_default_library,
    integrate_abd,
)

L, W, R, PLY_T, NX = 0.3, 0.15, 0.005, 0.125e-3, 1.0e5
LAMINATES = {
    "[0]8": [0] * 8,
    "[0/90]2s": [0, 90, 0, 90, 90, 0, 90, 0],
    "[0/45/-45/90]s": [0, 45, -45, 90, 90, -45, 45, 0],
    "[±45]2s": [45, -45, 45, -45, -45, 45, -45, 45],
}


def engineering_constants(lam):
    a = np.linalg.inv(integrate_abd(lam)[:3, :3]) * lam.thickness
    return 1 / a[0, 0], 1 / a[1, 1], 1 / a[2, 2], -a[0, 1] / a[0, 0]  # Ex, Ey, Gxy, nu_xy


def lekhnitskii(lam):
    ex, ey, gxy, nxy = engineering_constants(lam)
    kt_inf = 1 + np.sqrt(2 * (np.sqrt(ex / ey) - nxy) + ex / gxy)
    ratio = 1 - 2 * R / W
    return kt_inf, kt_inf * (2 + ratio**3) / (3 * ratio)


def konish_whitney(xi, kt_inf):
    """sigma_x / sigma_inf on the transverse line through the hole, xi = y / R >= 1."""
    return 1 + 0.5 * xi**-2 + 1.5 * xi**-4 - 0.5 * (kt_inf - 3) * (5 * xi**-6 - 7 * xi**-8)


def solve(gm):
    mesh = gm.mesh
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    bcs.fix(gm.nodes("left"), [DOF.UX]).fix([mesh.closest_node((0.0, W / 2))], [DOF.UY])
    right = sorted(gm.nodes("right"), key=lambda n: mesh.node_coordinates(n)[1])
    y = np.array([mesh.node_coordinates(n)[1] for n in right])
    share = np.zeros_like(y)
    share[:-1] += np.diff(y) / 2
    share[1:] += np.diff(y) / 2
    loads = LoadCase()
    for node, s in zip(right, share):
        loads.add_nodal_load(node, DOF.UX, NX * s)
    return ResultsExtractor(StructuralGlobalSolver().solve(mesh, bcs, loads))


def main():
    card = create_default_library().get("T300/5208")
    # The default curved Quad8: quadratic elements resolve the narrow peak of the stiff stacks
    # best (example 18 compares the element types)
    builder = GmshPlateBuilder(L, W, [CircularHole(L / 2, W / 2, R)], size=0.012, hole_size=0.0003, transition=6)
    results = {}
    print(f"{'laminate':<16}{'Kt FE':>8}{'Kt Lekhnitskii':>16}{'difference':>12}")
    for name, angles in LAMINATES.items():
        lam = Laminate.from_angles(card, angles, PLY_T)
        gm = builder.build(GmshMeshImporter(lam, FSDTModel()))
        ex = solve(gm)
        nodes = ex.nodes()
        index = {n: i for i, n in enumerate(nodes.node_ids)}
        # Nodes nearest to the transverse line x = L/2, from the hole edge outwards.
        targets = W / 2 + R * np.linspace(1.0, 5.0, 60)
        line = list(dict.fromkeys(gm.mesh.closest_node((L / 2, y)) for y in targets))
        xi = np.array([np.hypot(*(gm.mesh.node_coordinates(n)[:2] - [L / 2, W / 2])) / R for n in line])
        ratio = np.array([nodes.N[index[n], 0] / NX for n in line])
        kt_inf, kt = lekhnitskii(lam)
        print(f"{name:<16}{ratio[0]:8.2f}{kt:16.2f}{(ratio[0] / kt - 1) * 100:11.1f}%")
        results[name] = dict(gm=gm, ex=ex, xi=xi, ratio=ratio, kt_inf=kt_inf, kt=kt)

    fig = plt.figure(figsize=(13.0, 4.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.9])
    ax0, ax1, ax2 = (fig.add_subplot(grid[0, i]) for i in range(3))

    show = results["[0/45/-45/90]s"]
    elements = show["ex"].elements()
    order = {4: [0, 1, 2, 3], 8: [0, 4, 1, 5, 2, 6, 3, 7]}
    quads = [e.coordinates[order[len(e.node_ids)], :2] * 1e3 for e in show["gm"].mesh.elements]
    patches = PolyCollection(quads, array=elements.N[:, 0] / NX, cmap=SEQUENTIAL, edgecolors="face", linewidths=0.2)
    ax0.add_collection(patches)
    ax0.add_collection(PolyCollection(quads, facecolors="none", edgecolors="#0b0b0b", linewidths=0.15, alpha=0.35))
    fig.colorbar(patches, ax=ax0, shrink=0.85, label="Nx / applied Nx (element)")
    zoom = 4 * R * 1e3
    ax0.set_xlim(L / 2 * 1e3 - zoom, L / 2 * 1e3 + zoom)
    ax0.set_ylim(W / 2 * 1e3 - zoom, W / 2 * 1e3 + zoom)
    ax0.set_aspect("equal")
    ax0.grid(False)
    ax0.set_xlabel("x [mm]")
    ax0.set_ylabel("y [mm]")
    ax0.set_title(f"Gmsh quads, {len(quads)} elements")

    xi_ref = np.linspace(1.0, 3.0, 200)
    for colour, (name, r) in zip(SERIES, results.items()):
        keep = r["xi"] <= 3.0
        ax1.plot(r["xi"][keep], r["ratio"][keep], "o", ms=4, color=colour, label=name)
        ax1.plot(xi_ref, konish_whitney(xi_ref, r["kt_inf"]), color=colour, lw=1.2, ls=(0, (4, 2)))
    ax1.plot([], [], color=REFERENCE, lw=1.2, ls=(0, (4, 2)), label="Konish-Whitney")
    ax1.axhline(1.0, color=TEXT_SECONDARY, lw=0.8)
    ax1.set_xlim(1.0, 3.0)
    ax1.set_xlabel("distance from the hole centre, y / R")
    ax1.set_ylabel("σx / σ∞ on the line x = L/2")
    ax1.set_title("Stiff-axis laminates concentrate stress most")
    ax1.legend(loc="upper right")

    names = list(results)
    positions = np.arange(len(names))
    fe = [results[n]["ratio"][0] for n in names]
    theory = [results[n]["kt"] for n in names]
    ax2.bar(positions - 0.18, fe, width=0.34, color=SERIES[0], label="FE (Gmsh mesh)")
    ax2.bar(positions + 0.18, theory, width=0.34, color=TEXT_SECONDARY, label="Lekhnitskii + width correction")
    for p, v in zip(positions, fe):
        ax2.text(p - 0.18, v + 0.1, f"{v:.1f}", ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(names, rotation=20)
    ax2.set_ylabel("Kt at the hole edge")
    worst = max(abs(r["ratio"][0] / r["kt"] - 1) for r in results.values())
    ax2.set_title(f"FE Kt within {100 * worst:.0f}% of Lekhnitskii")
    ax2.grid(axis="x", visible=False)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right")
    fig.text(0.0, -0.04, f"T300/5208, 8 plies of {PLY_T * 1e3:.3f} mm, {L * 1e3:.0f} x {W * 1e3:.0f} mm plate, hole d = "
             f"{2 * R * 1e3:.0f} mm, uniform Nx on the right edge, FSDT. Kt read from averaged nodal Nx.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "open_hole_gmsh.png")


if __name__ == "__main__":
    main()
