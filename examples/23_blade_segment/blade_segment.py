"""A sandwich-panel blade segment under aerodynamic, gravity and outboard loads, checked against a beam.

An 8 m prismatic segment of a cambered NACA 4421 blade (1.5 m chord) has foam-cored
sandwich panels at the leading and trailing edges, glass UD spar caps and two
monolithic shear webs: the layout of most wind and tidal turbine blades. It is
loaded the way a segment cut from a blade is loaded:

* suction and pressure on the two surfaces (a flap load with its own pitching moment),
* edgewise gravity (the blade horizontal, at 90 degrees azimuth),
* the shear force and torque of the outboard blade, put on a reference node at the
  tip that is tied to the tip section by a distributing link (RBE3), so the end of
  the segment is loaded without being stiffened.

``beam_section_properties`` extracts the section stiffness and shear centre from the
same shell model, and a beam built from them is integrated along the span with the
same loads. The two agree away from the clamped root, where the shell's restrained
warping stiffens it; the shell also shows what a beam cannot, the panels bulging
between the webs. The model is written to ParaView (``vtk/``).

Left: the mid-span section, with its distortion relative to the rigid section motion.
Middle: flap and edge deflection and twist along the span, shell against beam.
Right: the worst Tsai-Wu index in each region outside the root boundary layer,
faces and core apart.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from composite_laminate_model import (
    Airfoil,
    BladeBuilder,
    BoundaryConditions,
    ElasticProperties3D,
    ExactStressRecoverer,
    FSDTModel,
    HygrothermalProperties,
    Laminate,
    LoadCase,
    MaterialCard,
    Ply,
    ShearWeb,
    SkinRegion,
    Station,
    StrengthProperties3D,
    StructuralGlobalSolver,
    TsaiWuCriterion,
    add_reference_node,
    beam_section_properties,
    create_default_library,
    distributing_link,
    write_vtu,
)

CHORD, LENGTH, N_SPAN, PITCH = 1.5, 8.0, 24, 0.3
REGIONS = [SkinRegion("le_panel", 0.0, 0.15, 5), SkinRegion("spar_cap", 0.15, 0.45, 6),
           SkinRegion("te_panel", 0.45, 1.0, 10)]
WEBS = [ShearWeb("web_fwd", 0.15, 4), ShearWeb("web_aft", 0.45, 4)]
Q_SUCTION, Q_PRESSURE = 3.0e3, 1.5e3  # [Pa]
TIP_FORCE = (4.0e3, 15.0e3, 0.0)  # outboard edge and flap shear [N]
TIP_TORQUE = -3.0e3  # nose-down pitching moment of the outboard blade [N m]
GRAVITY = (-9.81, 0.0, 0.0)  # edgewise, towards the leading edge

FOAM = MaterialCard(
    name="PVC foam H80",
    elastic=ElasticProperties3D.isotropic(85e6, 0.3),
    strength=StrengthProperties3D(2.2e6, 1.4e6, 2.2e6, 1.4e6, 1.15e6),
    hygrothermal=HygrothermalProperties(alpha1=35e-6, alpha2=35e-6),
    density=80.0,
    description="Cross-linked PVC foam core (typical datasheet values)",
)


def laminates(glass):
    t = 0.5e-3
    face_out = [Ply(glass, t, a) for a in (45, -45, 0)]
    face_in = [Ply(glass, t, a) for a in (0, -45, 45)]
    sandwich = lambda core: Laminate(face_out + [Ply(FOAM, core, 0)] + face_in, name=f"sandwich {core * 1e3:.0f} mm")  # noqa: E731
    cap = Laminate([Ply(glass, t, a) for a in (45, -45)] + [Ply(glass, 1.0e-3, 0)] * 16
                   + [Ply(glass, t, a) for a in (-45, 45)], name="cap")
    web = Laminate.from_angles(glass, [45, -45] * 4, 0.5e-3, name="web")
    return {"le_panel": sandwich(10e-3), "te_panel": sandwich(15e-3), "spar_cap": cap, "web_fwd": web, "web_aft": web}


def section_loads(mesh, blade, load_elements):
    """Distributed force [fx, fy] and torque mz per unit length about the pitch axis."""
    f, mz = np.zeros(2), 0.0
    for e in mesh.shell_elements:
        rho_a = sum(p.thickness * p.material.density for p in e.laminate.plies)
        force = rho_a * e.area * np.asarray(GRAVITY)
        force = force + load_elements.get(e.element_id, 0.0) * e.area * np.asarray(e.frame)[2]
        c = np.asarray(e.centroid)
        f += force[:2]
        mz += c[0] * force[1] - c[1] * force[0]
    return f / LENGTH, mz / LENGTH


def beam_response(props, f, mz, z):
    """Integrate the beam from the clamped root: flap (y) and edge (x) deflection and twist."""
    C = np.linalg.inv(props.stiffness)
    xs, ys = props.shear_centre
    a = LENGTH - z
    vx, vy = f[0] * a + TIP_FORCE[0], f[1] * a + TIP_FORCE[1]
    mx = -f[1] * a**2 / 2 - TIP_FORCE[1] * a
    my = f[0] * a**2 / 2 + TIP_FORCE[0] * a
    mt = mz * a + TIP_TORQUE
    mt_sc = mt - (xs * vy - ys * vx)  # torque about the shear centre twists the section
    k = np.array([C @ np.array([0.0, Mx, My, T]) for Mx, My, T in zip(mx, my, mt_sc)])
    cumint = lambda g: np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(z))])  # noqa: E731
    th_x, th_y, twist = cumint(k[:, 1]), cumint(k[:, 2]), cumint(k[:, 3])
    return cumint(th_y), cumint(-th_x), twist  # u_x' = theta_y, u_y' = -theta_x


def main():
    glass = create_default_library().get("E-glass/epoxy")
    stations = [Station(0.0, CHORD, Airfoil.naca4("4421"), pitch_axis=PITCH),
                Station(LENGTH, CHORD, Airfoil.naca4("4421"), pitch_axis=PITCH)]
    blade = BladeBuilder(stations, REGIONS, WEBS, n_span=N_SPAN).build(laminates(glass), FSDTModel())
    props = beam_section_properties(blade)
    print(f"EI flap {props.EI_x:.4g} N m2, EI edge {props.EI_y:.4g} N m2, GJ {props.GJ:.4g} N m2, "
          f"shear centre {np.round(props.shear_centre, 3)} m, {props.mass_per_length:.1f} kg/m")

    mesh = blade.mesh
    bcs = BoundaryConditions().clamp(blade.nodes("root"))
    tip = add_reference_node(mesh, (0.0, 0.0, LENGTH))
    distributing_link(mesh, bcs, tip, blade.nodes("tip"))
    sections = sorted(blade.span_positions)[1:]
    probes = []
    for z in sections[:-1]:  # unloaded distributing links only measure the mean section motion
        probes.append(add_reference_node(mesh, (0.0, 0.0, z)))
        distributing_link(mesh, bcs, probes[-1], blade.section_nodes(z))
    probes.append(tip)

    pressures = {e: Q_SUCTION for r in ("le_panel_ss", "spar_cap_ss", "te_panel_ss") for e in blade.elements(r)}
    pressures.update({e: -Q_PRESSURE for r in ("le_panel_ps", "spar_cap_ps", "te_panel_ps") for e in blade.elements(r)})
    load = LoadCase("operating").add_body_load(GRAVITY)
    for e, p in pressures.items():
        load.add_pressure(p, [e])
    for dof, value in enumerate(TIP_FORCE):
        load.add_nodal_load(tip, dof, value)
    load.add_nodal_load(tip, 5, TIP_TORQUE)
    solution = StructuralGlobalSolver().solve(mesh, bcs, load)

    z = np.array([0.0] + sections)
    shell = np.array([np.zeros(6)] + [solution.node_displacement(p) for p in probes])
    f, mz = section_loads(mesh, blade, pressures)
    beam_x, beam_y, beam_twist = beam_response(props, f, mz, z)
    print(f"distributed load {f[0]:.0f}, {f[1]:.0f} N/m and {mz:.0f} N m/m about the pitch axis")
    print(f"tip flap deflection: shell {shell[-1, 1] * 1e3:.1f} mm, beam {beam_y[-1] * 1e3:.1f} mm; "
          f"edge: shell {shell[-1, 0] * 1e3:.1f} mm, beam {beam_x[-1] * 1e3:.1f} mm; "
          f"twist: shell {np.degrees(shell[-1, 5]):.3f} deg, beam {np.degrees(beam_twist[-1]):.3f} deg")

    criterion, recoverer = TsaiWuCriterion(), ExactStressRecoverer(mesh.hygrothermal_engine)
    regions = {"LE panel": "le_panel", "spar caps": "spar_cap", "TE panel": "te_panel", "webs": "webs"}
    worst = {}
    for label, name in regions.items():
        for eid in blade.elements(name):
            e = mesh.element(eid)
            for point in recoverer.ply_midplane_states(e, solution.element_displacements(eid)):
                ply = e.laminate.plies[point.ply_index]
                # the first chord from the clamp is a boundary layer (restrained warping): kept apart
                zone = "root" if e.centroid[2] < CHORD else "span"
                key = (label, "core" if ply.material is FOAM else "faces", zone)
                index = criterion.evaluate(point.material_stress_3d(), ply.material.strength).index
                if float(index) > worst.get(key, (0.0, None))[0]:
                    worst[key] = (float(index), e.centroid)
    for (label, part, zone), (value, where) in sorted(worst.items()):
        print(f"  {label:<10} {part:<6} {zone:<5} max Tsai-Wu index {value:.3f} at x = {where[0]:.2f} m, z = {where[2]:.2f} m")
    worst = {k: v[0] for k, v in worst.items()}

    out = Path(__file__).resolve().parent / "vtk"
    files = write_vtu(out / "blade_segment", solution, criterion, layered=True, thickness_scale=3.0)
    print("ParaView files:", ", ".join(p.name for p in files.values()))

    # ---------------------------------------------------------------- figure
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    mid = len(probes) // 2
    z_mid, p_mid = sections[mid], solution.node_displacement(probes[mid])
    scale = 30.0
    colours = {"le_panel": SERIES[0], "spar_cap": SERIES[1], "te_panel": SERIES[0], "web_fwd": SERIES[2],
               "web_aft": SERIES[2]}
    for name, colour in colours.items():
        for eid in blade.elements(name):
            nodes = [n for n in mesh.element(eid).node_ids[:4] if abs(mesh.node_coordinates(n)[2] - z_mid) < 1e-9]
            if len(nodes) != 2:
                continue
            X = np.array([mesh.node_coordinates(n) for n in nodes])
            U = np.array([solution.node_displacement(n)[:3] for n in nodes])
            # distortion: displacement less the rigid motion of the section at the probe
            rigid = p_mid[:3] + np.cross(p_mid[3:], X - np.array([0.0, 0.0, z_mid]))
            D = X + scale * (U - rigid)
            ax0.plot(X[:, 0], X[:, 1], color=TEXT_SECONDARY, lw=0.8, ls=(0, (2, 2)))
            ax0.plot(D[:, 0], D[:, 1], color=colour, lw=2.4 if name == "spar_cap" else 1.6, solid_capstyle="butt")
    ax0.plot([], [], color=SERIES[0], lw=1.6, label="sandwich panels")
    ax0.plot([], [], color=SERIES[1], lw=2.4, label="spar caps")
    ax0.plot([], [], color=SERIES[2], lw=1.6, label="shear webs")
    ax0.plot([], [], color=TEXT_SECONDARY, lw=0.8, ls=(0, (2, 2)), label="undeformed")
    ax0.plot(*props.shear_centre, "o", color=SERIES[3], mfc="none", mew=1.8, ms=7, label="shear centre")
    ax0.plot(0, 0, "+", color=REFERENCE, ms=9, mew=1.8, label="pitch axis")
    ax0.set_aspect("equal")
    ax0.set_xlabel("x, chordwise [m]")
    ax0.set_ylabel("y [m]")
    ax0.set_title(f"Panels bulge between the webs (x{scale:g}, z = {z_mid:.1f} m)", loc="left")
    ax0.legend(loc="lower center", ncol=3, fontsize=7.5)
    ax0.set_ylim(-0.55, 0.4)

    ax1.plot(z, shell[:, 1] * 1e3, color=SERIES[0], label="flap, shell")
    ax1.plot(z, beam_y * 1e3, color=REFERENCE, ls=(0, (4, 2)), lw=1.2, label="beam")
    ax1.plot(z, shell[:, 0] * 1e3, color=SERIES[1], label="edge, shell")
    ax1.plot(z, beam_x * 1e3, color=REFERENCE, ls=(0, (4, 2)), lw=1.2)
    ax1.set_xlabel("span z [m]")
    ax1.set_ylabel("deflection [mm]")
    ax1.set_title(f"Shell and beam agree (tip twist {np.degrees(shell[-1, 5]):.2f}° "
                  f"vs {np.degrees(beam_twist[-1]):.2f}°)", loc="left", fontsize=10)
    ax1.legend(loc="upper left")

    labels = list(regions)
    x = np.arange(len(labels))
    faces = [worst.get((l, "faces", "span"), 0.0) for l in labels]
    core = [worst.get((l, "core", "span"), np.nan) for l in labels]
    ax2.bar(x - 0.18, faces, width=0.34, color=SERIES[0], label="glass plies")
    ax2.bar(x + 0.18, core, width=0.34, color=SERIES[3], label="foam core")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("max Tsai-Wu index")
    ax2.set_title(f"Worst index in each region (z > {CHORD:g} m)", loc="left")
    ax2.grid(axis="x", visible=False)
    ax2.set_ylim(0.0, 1.3 * np.nanmax(faces + core))
    ax2.legend(loc="upper left", ncol=2)
    fig.text(0.0, -0.05, f"NACA 4421, {CHORD:g} m chord, {LENGTH:g} m, curved Quad8 FSDT, {N_SPAN} elements along the span. "
             f"E-glass/epoxy faces [±45/0], PVC foam 10 mm (LE) and 15 mm (TE), caps [±45/16 x 1 mm UD/∓45], "
             f"webs [±45]4. Suction {Q_SUCTION / 1e3:g} kPa, pressure {Q_PRESSURE / 1e3:g} kPa, edgewise gravity, "
             f"tip shear ({TIP_FORCE[0] / 1e3:g}, {TIP_FORCE[1] / 1e3:g}) kN and {TIP_TORQUE / 1e3:g} kN m through RBE3.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "blade_segment.png")


if __name__ == "__main__":
    main()
