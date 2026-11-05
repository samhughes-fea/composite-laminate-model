"""Spar caps merged into the skin, or separate shells tied to it.

A spar cap is laid on the inside of the skin, so the cap and the skin have separate
mid-surfaces. ``BladeBuilder.from_geometry`` can mesh it two ways:

- ``caps="merged"`` (the default): along the cap the shell is the cap and skin
  stacked, on the stack's mid-surface. The skin's reference surface steps in by half
  the cap thickness at each cap edge.
- ``caps="tied"``: the skin stays on its own mid-surface, the cap is a shell of its own
  on its mid-surface, and every cap node is tied to the skin node facing it by a rigid
  offset link (RBE2). This is blade-structure's section shell model
  (``cap_tie_mode="skin_offset"``).

This example meshes one blade segment both ways and compares:

1. the beam stiffness and shear centre (``beam_section_properties``);
2. how a flapwise moment and a torque share their membrane loads between the cap and
   the skin beneath it, against the split a stacked laminate makes (each layer's
   share of A11 for the axial load and of A66 for the shear).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from composite_laminate_model import (
    BladeBuilder,
    BoundaryConditions,
    FSDTModel,
    Laminate,
    LoadCase,
    ResultsExtractor,
    StructuralGlobalSolver,
    beam_section_properties,
    create_default_library,
    integrate_abd,
)
from composite_laminate_model.laminate_connectors import add_reference_node, rigid_link
from implicit_geometry_kernel import Airfoil, BladeGeometry, Station

SPAN, CHORD = 5.0, 1.0
T_SKIN, T_CAP, T_WEB = 0.003, 0.008, 0.002
COLOURS = {"merged": SERIES[0], "tied": SERIES[1]}


def model():
    airfoil = Airfoil.naca4("0024")
    stations = [Station(0.0, CHORD, airfoil, pitch_axis=0.3), Station(SPAN, CHORD, airfoil, pitch_axis=0.3)]
    geometry = BladeGeometry(stations, "2D-F", T_SKIN, spar_cap_thickness=T_CAP, shear_web_thickness=T_WEB,
                             web_positions=(0.18, 0.48))
    lib = create_default_library()
    glass, carbon = lib.get("E-glass/epoxy"), lib.get("T300/5208")
    laminates = {
        "skin": Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3, "skin"),
        "web": Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3, "web"),
        "cap": Laminate.from_angles(carbon, [0, 0, 0, 45, -45, 0, 0, 0, 0, 0, 0, -45, 45, 0, 0, 0], 0.5e-3, "cap"),
    }
    return geometry, laminates


def build(geometry, laminates, caps):
    builder = BladeBuilder.from_geometry(geometry, n_span=20, element_size=0.02, caps=caps)
    return builder.build(laminates, FSDTModel())


def cantilever(blade, load):
    """Clamped root, rigid tip on a node at the pitch axis carrying ``load`` (dof, value) pairs."""
    mesh = blade.mesh.copy()
    bcs = blade.tie_caps(BoundaryConditions(), mesh).clamp(blade.nodes("root"))
    tip = add_reference_node(mesh, (0.0, 0.0, SPAN))
    rigid_link(mesh, bcs, tip, blade.nodes("tip"))
    case = LoadCase()
    for dof, value in load:
        case.add_nodal_load(tip, dof, value)
    return ResultsExtractor(StructuralGlobalSolver().solve(mesh, bcs, case)).elements()


def cap_shares(blade, field, component):
    """|N| of each cap element over cap + the skin element under it, in the middle element row."""
    ids = {int(e): k for k, e in enumerate(field.element_ids)}
    z = field.centroid[:, 2]
    row = np.abs(z - 0.5 * SPAN) <= 0.5 * SPAN / 20 + 1e-9
    caps = [ids[e] for e in blade.elements("spar_cap_suction_surface") if row[ids[e]]]
    skins = np.array([ids[e] for e in blade.elements("skin") if row[ids[e]]])
    x, share = [], []
    for k in caps:
        j = skins[np.argmin(np.linalg.norm(field.centroid[skins] - field.centroid[k], axis=1))]
        a, b = abs(field.N[k, component]), abs(field.N[j, component])
        x.append(field.centroid[k, 0])
        share.append(a / (a + b))
    order = np.argsort(x)
    return np.array(x)[order], np.array(share)[order]


def main():
    geometry, laminates = model()
    blades = {caps: build(geometry, laminates, caps) for caps in ("merged", "tied")}
    for caps, blade in blades.items():
        print(f"{caps}: {blade.mesh.n_nodes} nodes, {blade.mesh.n_elements} elements, {len(blade.ties)} links")

    # 1. beam stiffness
    props = {caps: beam_section_properties(b, end_ribs=0.2) for caps, b in blades.items()}
    names = ("EA", "EI_x", "EI_y", "GJ")
    print("\n       " + "".join(f"{n:>12}" for n in names) + "   shear centre x (mm)")
    for caps, p in props.items():
        print(f"{caps:7}" + "".join(f"{getattr(p, n):12.4g}" for n in names)
              + f"   {p.shear_centre[0] * 1e3:8.2f}")
    ratio = [getattr(props["tied"], n) / getattr(props["merged"], n) for n in names]
    print("tied / merged " + "  ".join(f"{n} {r:.4f}" for n, r in zip(names, ratio)))

    # 2. load sharing between cap and skin at mid-span
    A = {k: integrate_abd(laminates[k])[:3, :3] for k in ("skin", "cap")}
    split = {"Nx": A["cap"][0, 0] / (A["cap"][0, 0] + A["skin"][0, 0]),
             "Nxy": A["cap"][2, 2] / (A["cap"][2, 2] + A["skin"][2, 2])}
    tied = blades["tied"]
    bending = cantilever(tied, [(3, 1.0e4)])  # flapwise moment Mx
    torsion = cantilever(tied, [(5, 1.0e3)])  # torque Mz
    xb, sb = cap_shares(tied, bending, 0)
    xt, st = cap_shares(tied, torsion, 2)
    print(f"\ncap share of Nx under bending: tied median {np.median(sb):.3f}, stacked split {split['Nx']:.3f}")
    print(f"cap share of Nxy under torsion: tied median {np.median(st):.3f}, stacked split {split['Nxy']:.3f}")

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1.0]})
    ax = axes[0]
    z = 0.5 * SPAN
    cap_label = "spar_cap_suction_surface"
    edge = None
    for caps, blade in blades.items():
        mesh = blade.mesh
        nodes = blade.section_nodes(z) + [c for c in blade.ties if abs(mesh.node_coordinates(c)[2] - z) < 1e-9]
        xy = np.array([mesh.node_coordinates(n)[:2] for n in nodes])
        ax.plot(xy[:, 0], xy[:, 1], "o", ms=3.2, color=COLOURS[caps], label=f"{caps} nodes", zorder=3)
        if caps == "tied":
            for c, s in blade.ties.items():
                pc, ps = mesh.node_coordinates(c), mesh.node_coordinates(s)
                if abs(pc[2] - z) < 1e-9:
                    ax.plot([pc[0], ps[0]], [pc[1], ps[1]], color=COLOURS[caps], lw=0.8)
            cap_nodes = [c for c in blade.ties if abs(mesh.node_coordinates(c)[2] - z) < 1e-9]
            cap_xy = np.array([mesh.node_coordinates(c)[:2] for c in cap_nodes])
            top = cap_xy[cap_xy[:, 1] > 0]
            edge = top[np.argmin(top[:, 0])]
    cut = geometry.section_at(z)
    for w in cut.walls:
        if w.kind == "skin" and w.side == "suction":
            ax.plot(w.xy[:, 0], w.xy[:, 1] + 0.5 * w.thickness, color=REFERENCE, lw=0.8, ls="--")
    ax.set_xlim(edge[0] - 0.05, edge[0] + 0.07)
    ax.set_ylim(edge[1] - 0.02, edge[1] + 0.03)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Forward edge of the suction-side cap, mid-span")
    ax.plot([], [], color=REFERENCE, lw=0.8, ls="--", label="outer mould line (approx.)")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(xb, sb, "o-", ms=3, color=SERIES[1], label="tied, Nx under flapwise moment")
    ax.plot(xt, st, "s-", ms=3, color=SERIES[2], label="tied, Nxy under torque")
    ax.axhline(split["Nx"], color=SERIES[1], ls="--", lw=1.2, label="stacked split, A11")
    ax.axhline(split["Nxy"], color=SERIES[2], ls="--", lw=1.2, label="stacked split, A66")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("x along the cap (m)")
    ax.set_ylabel("cap share of cap + skin")
    ax.set_title("Load shared with the skin beneath")
    ax.legend(loc="lower center", fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "tied_spar_caps.png")

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(names, [100.0 * (r - 1.0) for r in ratio], color=SERIES[1])
    ax.axhline(0.0, color=TEXT_SECONDARY, lw=0.8)
    ax.set_ylabel("tied - merged (%)")
    ax.set_title("Beam stiffness, tied caps against merged")
    ax.grid(axis="x", visible=False)
    for k, r in enumerate(ratio):
        ax.annotate(f"{100 * (r - 1):+.2f}%", (k, 100.0 * (r - 1.0)), ha="center",
                    va="bottom" if r >= 1 else "top", fontsize=8.5)
    ax.set_ylim(min(-2.5, 100 * (min(ratio) - 1) - 0.8), 1.0)
    fig.tight_layout()
    save(fig, __file__, "tied_spar_caps_stiffness.png")


if __name__ == "__main__":
    main()
