"""A tapered blade segment whose spar caps are defined ply by ply, with ply drops.

The segment runs 10 m from a 1.2 m NACA 0024 root to a 0.6 m NACA 0018 tip with 6
degrees of twist. Skins, webs and the spar caps come from ``BladeBuilder``; the caps
are then rebuilt from a ply book (``Layup``): skin plies run over the whole cap, eight
unidirectional cap plies run the full length and twelve more stop in pairs between
2 m and 7 m, the way a real cap tapers towards the tip.

A flap pressure that grows linearly along the span loads the pressure side. The
ply book reports where every ply stops, checks that no more than two plies stop on
one edge, and gives the highest Tsai-Wu index of each named ply. The model is also
written to ParaView (``vtk/``) with one cell per ply.

Left: cap thickness along the span. Middle: spanwise strain on the suction-side cap
with the tip deflection in the title. Right: the worst Tsai-Wu index of each cap ply.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import (
    Airfoil,
    BladeBuilder,
    BoundaryConditions,
    FSDTModel,
    Laminate,
    Layup,
    LoadCase,
    PlyDefinition,
    ShearWeb,
    SkinRegion,
    Station,
    StructuralGlobalSolver,
    TsaiWuCriterion,
    create_default_library,
    write_vtu,
)

LENGTH, N_SPAN = 10.0, 40
FULL_PLIES, DROPS = 8, (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)  # two plies stop at each drop station
Q_TIP = 3.0e3  # flap pressure at the tip [Pa], zero at the root
REGIONS = [SkinRegion("le_panel", 0.0, 0.2, 4), SkinRegion("spar_cap", 0.2, 0.5, 6), SkinRegion("te_panel", 0.5, 1.0, 8)]
WEBS = [ShearWeb("web_fwd", 0.2, 4), ShearWeb("web_aft", 0.5, 4)]


def ply_book(glass):
    """Inner skin, cap UD plies (some dropping), outer skin; listed from the inside out."""
    t_skin, t_ud = 0.5e-3, 1.0e-3
    plies = [PlyDefinition(f"skin_in_{k}", glass, t_skin, a, elements="spar_cap") for k, a in enumerate((45, -45))]
    ends = [None] * FULL_PLIES + [z for z in DROPS for _ in range(2)]
    for k, z_end in enumerate(ends):
        region = None if z_end is None else (lambda x, z_end=z_end: x[2] < z_end)
        plies.append(PlyDefinition(f"cap_{k + 1:02d}", glass, t_ud, 0.0, elements="spar_cap", region=region))
    plies += [PlyDefinition(f"skin_out_{k}", glass, t_skin, a, elements="spar_cap") for k, a in enumerate((-45, 45, 0))]
    return Layup(plies, name="cap", reference="top")  # outer mould line fixed, as laid in the mould


def main():
    glass = create_default_library().get("E-glass/epoxy")
    skin = Laminate.from_angles(glass, [45, -45, 0, -45, 45], 0.5e-3)
    web = Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3)
    stations = [Station(0.0, 1.2, Airfoil.naca4("0024"), twist=0.0, pitch_axis=0.35),
                Station(LENGTH, 0.6, Airfoil.naca4("0018"), twist=-6.0, pitch_axis=0.35)]
    blade = BladeBuilder(stations, REGIONS, WEBS, n_span=N_SPAN).build(
        {"le_panel": skin, "te_panel": skin, "spar_cap": skin, "web_fwd": web, "web_aft": web}, FSDTModel())
    mesh = blade.mesh
    layup = ply_book(glass)
    caps = blade.elements("spar_cap")
    layup.apply(mesh, blade.element_sets, element_ids=caps)

    drops = layup.drops(mesh, blade.element_sets)
    print(f"{len(drops)} ply-drop edges; plies stopping: {sorted({d.ply for d in drops})}")
    issues = layup.check_drops(mesh, max_per_step=2, element_sets=blade.element_sets)
    print("ply-drop check:", "pass (no more than 2 plies stop on one edge)" if not issues else issues[:3])
    mass = sum(r["mass"] for r in layup.ply_table(mesh, blade.element_sets))
    print(f"spar cap mass from the ply book: {mass:.1f} kg")

    bcs = BoundaryConditions().clamp(blade.nodes("root"))
    load = LoadCase("flap").add_pressure(lambda x: -Q_TIP * x[2] / LENGTH, blade.elements("le_panel_ps")
                                         + blade.elements("spar_cap_ps") + blade.elements("te_panel_ps"))
    solution = StructuralGlobalSolver().solve(mesh, bcs, load)
    criterion = TsaiWuCriterion()
    worst = layup.ply_failure(solution, criterion)
    tip = np.mean([solution.node_displacement(n)[1] for n in blade.nodes("tip")])
    print(f"tip flap deflection {tip * 1e3:.1f} mm")
    name, w = max(((n, v) for n, v in worst.items() if n in {p.name for p in layup.plies}), key=lambda kv: kv[1]["index"])
    print(f"highest Tsai-Wu index in the caps {w['index']:.3f} in ply {name}, element {w['element']} "
          f"(z = {mesh.element(w['element']).centroid[2]:.2f} m)")

    out = Path(__file__).resolve().parent / "vtk"
    files = write_vtu(out / "blade", solution, criterion, layered=True, thickness_scale=5.0)
    print("ParaView files:", ", ".join(p.name for p in files.values()))

    # ---------------------------------------------------------------- figure
    ss = sorted(blade.elements("spar_cap_ss"), key=lambda e: mesh.element(e).centroid[2])
    z = np.array([mesh.element(e).centroid[2] for e in ss])
    h = np.array([mesh.element(e).laminate.thickness for e in ss])
    order = np.argsort(z)
    z, h, ss = z[order], h[order], [ss[i] for i in order]
    # one value per spanwise station (mean over the chordwise cap elements)
    zs = np.unique(np.round(z, 6))
    hs = np.array([h[np.isclose(z, v)].mean() for v in zs])
    strain = []
    for e in ss:
        el = mesh.element(e)
        u = solution.element_displacements(e)
        # spanwise (local x = material axis) membrane strain at the cap mid-plane, mean of the 2 x 2 Gauss points
        g = 1.0 / np.sqrt(3.0)
        zm = el.laminate.mid_plane  # the reference surface is the outer face; strain at the cap's own mid-plane
        strain.append(np.mean([(lambda e: e[0] + zm * e[3])(el.strain_state(u, a, b)[0]) for a in (-g, g) for b in (-g, g)]))
    strain = np.array(strain)
    eps_s = np.array([strain[np.isclose(z, v)].mean() for v in zs])
    sections = sorted(blade.span_positions)
    defl = [np.mean([solution.node_displacement(n)[1] for n in blade.section_nodes(zz)]) for zz in sections]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.0, 1.1, 1.3]})
    ax0.step(zs, hs * 1e3, where="mid", color=SERIES[0])
    for d in DROPS:
        ax0.axvline(d, color=TEXT_SECONDARY, lw=0.6, ls=(0, (2, 2)))
    ax0.set_xlabel("span z [m]")
    ax0.set_ylabel("cap laminate thickness [mm]")
    ax0.set_title("Two plies stop at each station", loc="left")

    ax1.plot(zs, eps_s * 1e6, color=SERIES[1])
    for d in DROPS:
        ax1.axvline(d, color=TEXT_SECONDARY, lw=0.6, ls=(0, (2, 2)))
    ax1.set_xlabel("span z [m]")
    ax1.set_ylabel("spanwise strain at the cap mid-plane [µε]")
    ax1.set_title(f"Strain steps up at each drop (tip {defl[-1] * 1e3:.0f} mm)", loc="left")

    names = [p.name for p in layup.plies if p.name.startswith("cap_")]
    values = [worst[n]["index"] for n in names]
    colours = [SERIES[0] if i < FULL_PLIES else SERIES[3] for i in range(len(names))]
    ax2.bar(range(len(names)), values, color=colours, width=0.7)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.split("_")[1] for n in names], fontsize=7)
    ax2.set_xlabel("cap ply (blue: full length, yellow: dropped)")
    ax2.set_ylabel("max Tsai-Wu index")
    ax2.set_title("Worst index of each named ply", loc="left")
    ax2.grid(axis="x", visible=False)
    fig.text(0.0, -0.05, f"Root 1.2 m NACA 0024 to tip 0.6 m NACA 0018, −6° twist, {LENGTH:g} m, {N_SPAN} elements along the span, "
             f"curved Quad8 FSDT. E-glass/epoxy: skins [±45/0/∓45], caps [±45 / {FULL_PLIES}–20 x 1 mm UD / ∓45, 0], webs [±45]s. "
             f"Flap pressure 0 to {Q_TIP / 1e3:g} kPa on the pressure side; root clamped.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "spar_cap_ply_drops.png")


if __name__ == "__main__":
    main()
