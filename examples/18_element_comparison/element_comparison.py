"""Every shell element on the open-hole plate: accuracy per degree of freedom.

The quasi-isotropic open-hole plate of example 08 is meshed by Gmsh with each of
the element types at several sizes. The stress concentration at the hole edge is
read from averaged nodal Nx and compared with Lekhnitskii's solution (with the
finite-width correction).

Left: the three meshes at the second refinement, zoomed on the hole. Right: Kt
error against the number of degrees of freedom.

Needs Gmsh: pip install "composite-laminate-fea[gmsh]".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_open_hole_gmsh"))

import numpy as np
from matplotlib.collections import PolyCollection

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from open_hole_gmsh import L, PLY_T, R, W, lekhnitskii
from laminate_modelling_package import (
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
)

NX = 1.0e5
ELEMENTS = {"Tri3": "tri3", "Quad4": "quad4", "Quad8": "quad8", "Tri6": "tri6", "Curved Quad8": "quad8_curved"}
COLOURS = {"Tri3": SERIES[0], "Quad4": SERIES[1], "Quad8": SERIES[2], "Tri6": SERIES[3], "Curved Quad8": SERIES[4]}
SHOWN = ("Tri3", "Quad4", "Quad8")  # meshes drawn on the left (Tri6 and curved Quad8 share these layouts)
REFINEMENTS = [(0.024, 0.0016), (0.016, 0.0008), (0.012, 0.0004), (0.010, 0.0002)]  # (size, hole size)
POLYGON = {3: [0, 1, 2], 4: [0, 1, 2, 3], 6: [0, 3, 1, 4, 2, 5], 8: [0, 4, 1, 5, 2, 6, 3, 7]}


def kt(gm):
    mesh = gm.mesh
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    bcs.fix(gm.nodes("left"), [DOF.UX]).fix([mesh.closest_node((0.0, W / 2))], [DOF.UY])
    loads = LoadCase().add_line_load(gm.nodes("right"), (NX, 0.0, 0.0))
    result = StructuralGlobalSolver().solve(mesh, bcs, loads)
    nodes = ResultsExtractor(result).nodes()
    edge = mesh.closest_node((L / 2, W / 2 + R))
    return nodes.N[list(nodes.node_ids).index(edge), 0] / NX, mesh.n_dof


def main():
    card = create_default_library().get("T300/5208")
    lam = Laminate.from_angles(card, [0, 45, -45, 90, 90, -45, 45, 0], PLY_T)
    _, kt_ref = lekhnitskii(lam)
    curves = {name: [] for name in ELEMENTS}
    shown = {}
    print(f"Lekhnitskii Kt (finite width) = {kt_ref:.3f}")
    print(f"{'element':<14}{'size [mm]':>10}{'hole [mm]':>10}{'DOFs':>8}{'Kt':>8}{'error':>9}")
    for name, element in ELEMENTS.items():
        for i, (size, hole_size) in enumerate(REFINEMENTS):
            builder = GmshPlateBuilder(L, W, [CircularHole(L / 2, W / 2, R)], size=size, hole_size=hole_size,
                                       transition=6, element=element)
            gm = builder.build(GmshMeshImporter(lam, FSDTModel()))
            value, ndof = kt(gm)
            curves[name].append((ndof, value))
            print(f"{name:<14}{size * 1e3:10.1f}{hole_size * 1e3:10.2f}{ndof:8d}{value:8.3f}{100 * (value / kt_ref - 1):8.1f}%")
            if i == 1:
                shown[name] = gm

    fig = plt.figure(figsize=(13.0, 4.4))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5])
    zoom = 3.0 * R * 1e3
    for k, name in enumerate(SHOWN):
        ax = fig.add_subplot(grid[0, k])
        polys = [e.coordinates[POLYGON[len(e.node_ids)], :2] * 1e3 for e in shown[name].mesh.elements]
        ax.add_collection(PolyCollection(polys, facecolors="#f0efec", edgecolors=COLOURS[name], linewidths=0.5))
        ax.set_xlim(L / 2 * 1e3 - zoom, L / 2 * 1e3 + zoom)
        ax.set_ylim(W / 2 * 1e3 - zoom, W / 2 * 1e3 + zoom)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}, {shown[name].mesh.n_dof} DOFs", loc="left")
    ax = fig.add_subplot(grid[0, 3])
    for name, points in curves.items():
        ndof, values = np.array(points).T
        style = dict(ls=(0, (4, 2)), marker="s", mfc="none") if name in ("Tri6", "Curved Quad8") else dict(marker="o")
        ax.plot(ndof, 100 * np.abs(values / kt_ref - 1), color=COLOURS[name], label=name, **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("|Kt error| vs Lekhnitskii [%]")
    ax.set_title("Quadratic elements converge first; Tri3 lags", loc="left")
    ax.legend(loc="upper right")
    fig.text(0.0, -0.04, f"[0/45/-45/90]s T300/5208, {L * 1e3:.0f} x {W * 1e3:.0f} mm plate, hole d = {2 * R * 1e3:.0f} mm, "
             f"uniform Nx, FSDT, Gmsh meshes graded to the hole. Lekhnitskii Kt = {kt_ref:.2f}. Kt from averaged nodal Nx.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "element_comparison.png")


if __name__ == "__main__":
    main()
