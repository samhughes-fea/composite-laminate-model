"""Interlaminar stresses at a spar-cap ply drop: a layered-solid detail inside a shell model.

Where plies stop in a spar cap, the load in the dropped plies has to shear across
into the plies that continue. A shell model sees only a step in membrane stress;
the interlaminar shear and peel that start delaminations at the drop need a 3D
model. Here a 20 mm long strip of the cap around the drop is meshed with layered
Hex20 solids (one element per ply, graded towards the drop) and joined to shell
models of the thick and thin cap on either side with ``couple_shell_to_solid``.
Two solid blocks (10 plies and 8 plies) are joined with ``merge_coincident_nodes``.

The strip carries the cap's spanwise tension; it is gripped at both ends and held
in plane strain across its width, as inside a wide cap.

Left: interlaminar shear and peel along the interface under the two dropped plies.
Middle: shear through the thickness close to the drop and far from it. Right: the
interlaminar shear over the side of the strip.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib.tri import Triangulation

from _shared.plotting import DIVERGING, SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import (
    BoundaryConditions,
    FEAMesh,
    FSDTModel,
    Laminate,
    LoadCase,
    RectangularPlateBuilder,
    StructuralGlobalSolver,
    couple_shell_to_solid,
    create_default_library,
    merge_coincident_nodes,
    solid_block,
)

T_PLY, N_THICK, N_DROP = 1.0e-3, 10, 2
WIDTH, L_SOLID, L_SHELL = 10e-3, 10e-3, 60e-3
STRESS = 100e6  # far-field stress in the thin cap [Pa]


def graded(n, towards):
    """Node fractions 0..1 clustered towards 0 or 1."""
    f = 1.0 - np.cos(0.5 * np.pi * np.linspace(0.0, 1.0, n + 1))
    return f if towards == 0 else 1.0 - f[::-1]


def add_shells(mesh, lam, x0, length, nx):
    named = RectangularPlateBuilder(length, WIDTH, nx, 2, origin=(x0, 0.0, 0.5 * lam.thickness),
                                    element="quad8").build_named(lam, FSDTModel())
    off, eoff = max(mesh.node_ids), max(mesh.element_ids)
    for n in named.mesh.node_ids:
        mesh.add_node(n + off, named.mesh.node_coordinates(n))
    for e in named.mesh.elements:
        mesh.add_element(e.element_id + eoff, [n + off for n in e.node_ids], lam, FSDTModel())
    return {k: [n + off for n in v] for k, v in named.node_sets.items()}


def main():
    glass = create_default_library().get("E-glass/epoxy")
    thick = Laminate.from_angles(glass, [0] * N_THICK, T_PLY)
    thin = Laminate.from_angles(glass, [0] * (N_THICK - N_DROP), T_PLY)
    mesh = FEAMesh()
    # solid_block puts the laminate mid-plane on origin z, so offset each block to sit on z = 0
    a = solid_block(mesh, thick, L_SOLID, WIDTH, graded(14, 1), 2, origin=(-L_SOLID, 0, 0.5 * thick.thickness),
                    element="hex20")
    b = solid_block(mesh, thin, L_SOLID, WIDTH, graded(14, 0), 2, origin=(0, 0, 0.5 * thin.thickness), element="hex20")
    merged = merge_coincident_nodes(mesh, a, b)
    left = add_shells(mesh, thick, -L_SOLID - L_SHELL, L_SHELL, 12)
    right = add_shells(mesh, thin, L_SOLID, L_SHELL, 12)
    bcs = BoundaryConditions()
    couple_shell_to_solid(mesh, bcs, left["right"], a.nodes("x0"))
    couple_shell_to_solid(mesh, bcs, right["left"], b.nodes("x1"))
    constrained = {k[0] for k in bcs.constrained_keys()}
    bcs.fix([n for n in a.nodes("y0") + a.nodes("y1") + b.nodes("y0") + b.nodes("y1") if n not in constrained], [1])
    for s in (left, right):
        bcs.fix(s["bottom"] + s["top"], [1, 3, 5])  # plane strain across the width
    bcs.clamp(left["left"])
    bcs.fix(right["right"], [4])  # gripped: no end rotation
    load = LoadCase().add_line_load(right["right"], (STRESS * thin.thickness, 0.0, 0.0))
    sol = StructuralGlobalSolver().solve(mesh, bcs, load)
    print(f"{mesh.n_elements} elements ({len(mesh.solid_elements)} solids), {mesh.n_dof} DOFs, {len(merged)} nodes merged")

    z_int = (N_THICK - N_DROP) * T_PLY  # interface under the dropped plies
    line, field = [], []
    for e in mesh.solid_elements:
        u = sol.element_displacements(e.element_id)
        z_bot = e.centroid[2] - 0.5 * e.laminate.thickness
        for xi in (-0.775, 0.0, 0.775):
            for zeta in (-0.775, 0.0, 0.775):
                p = e.stress_at(u, xi, 0.0, zeta)
                field.append((p.xyz[0], p.xyz[2], p.stress_global[4]))
            if e.centroid[0] < 0 and abs(z_bot - z_int) < 1e-9:  # first dropped ply, on the interface
                p = e.stress_at(u, xi, 0.0, -1.0)
                line.append((p.xyz[0], p.stress_global[4], p.stress_global[2]))
    line = np.array(sorted(line))
    peak = line[np.argmax(np.abs(line[:, 1]))]
    print(f"peak interlaminar shear {peak[1] / 1e6:.1f} MPa and peel {line[-1, 2] / 1e6:.1f} MPa at the drop, "
          f"for {STRESS / 1e6:.0f} MPa in the thin cap; ILSS of glass/epoxy is roughly 40-70 MPa")

    def profile(x_target):
        rows = []
        for e in mesh.solid_elements:
            lo, hi = e.coordinates[:, 0].min(), e.coordinates[:, 0].max()
            if lo - 1e-12 <= x_target <= hi + 1e-12 and e.coordinates[:, 1].min() < 1e-12:
                xi = 2.0 * (x_target - lo) / (hi - lo) - 1.0
                u = sol.element_displacements(e.element_id)
                # the 2-point Gauss heights, where displacement-based shear is most accurate
                for zeta in (-0.57735, 0.57735):
                    p = e.stress_at(u, xi, 0.0, zeta)
                    rows.append((p.xyz[2], p.stress_global[4]))
        return np.array(sorted(rows))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.1, 0.8, 1.4]})
    ax0.plot(line[:, 0] * 1e3, line[:, 1] / 1e6, color=SERIES[0], label="τxz, interlaminar shear")
    ax0.plot(line[:, 0] * 1e3, line[:, 2] / 1e6, color=SERIES[1], label="σz, peel")
    ax0.axhline(0.0, color=TEXT_SECONDARY, lw=0.8, label="shell model (no interlaminar stress)")
    ax0.set_xlabel("distance from the drop x [mm]")
    ax0.set_ylabel("stress on the interface [MPa]")
    ax0.set_title("Shear and peel peak at the ply end", loc="left")
    ax0.legend(loc="upper left")

    for x_t, colour, label in ((-1e-3, SERIES[0], "1 mm behind the drop"), (-8e-3, SERIES[2], "8 mm behind")):
        pr = profile(x_t)
        ax1.plot(pr[:, 1] / 1e6, pr[:, 0] * 1e3, color=colour, label=label)
    ax1.axhline(z_int * 1e3, color=TEXT_SECONDARY, lw=0.6, ls=(0, (2, 2)))
    ax1.text(ax1.get_xlim()[1], z_int * 1e3 + 0.15, "dropped plies above ", fontsize=7.5, color=TEXT_SECONDARY, ha="right")
    ax1.set_xlabel("τxz [MPa]")
    ax1.set_ylabel("z [mm]")
    ax1.set_title("Through the thickness", loc="left")
    ax1.legend(loc="lower right")

    f = np.array(field)
    tri = Triangulation(f[:, 0] * 1e3, f[:, 1] * 1e3)
    cx = f[tri.triangles, 0].mean(axis=1)
    cz = f[tri.triangles, 1].mean(axis=1)
    tri.set_mask((cx > 0.0) & (cz > z_int))  # no material above the thin cap
    tc = ax2.tricontourf(tri, f[:, 2] / 1e6, levels=np.arange(-40.0, 40.1, 5.0), cmap=DIVERGING, extend="both")
    ax2.plot([-L_SOLID * 1e3, 0, 0, L_SOLID * 1e3], [N_THICK, N_THICK, N_THICK - N_DROP, N_THICK - N_DROP],
             color=TEXT_SECONDARY, lw=1.0)
    fig.colorbar(tc, ax=ax2, label="τxz [MPa]", ticks=np.arange(-40, 41, 10))
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("z [mm]")
    ax2.set_title("Shear transfer around the drop", loc="left")
    fig.text(0.0, -0.05, f"E-glass/epoxy UD cap, {N_THICK} → {N_THICK - N_DROP} plies x {T_PLY * 1e3:g} mm, plies end square "
             f"(no resin pocket). Solid detail ±{L_SOLID * 1e3:g} mm (Hex20, one element per ply, graded to the drop), "
             f"Quad8 shells {L_SHELL * 1e3:g} mm each side, {STRESS / 1e6:g} MPa in the thin cap, plane strain across the width.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "ply_drop_solid_detail.png")


if __name__ == "__main__":
    main()
