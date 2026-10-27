"""Section properties of the IEA 15 MW reference blade: shell model against PreComp.

The IEA Wind 15 MW reference turbine publishes its blade as a full layup (windIO
file): airfoils, chord, spar caps, webs, sandwich panels and trailing-edge
reinforcement, with materials. ``data/iea15mw_sections.json`` holds that layup at
six stations as PreComp sees it (sector boundaries on each surface, laminae per
sector, web positions), PreComp's section properties for each, and the published
stiffness and mass data of the turbine model. ``data/prepare_reference_data.py``
regenerates it with WISDEM.

Each station is built as a prismatic shell segment, eight chords long, with
``BladeBuilder`` (laminates growing inward from the outer mould line, meshed on
their mid-surface, webs node-to-node between the skins), and
``beam_section_properties`` extracts its beam properties. The comparison is like for
like: same airfoil points, same sector boundaries, same laminae, same material
constants.

The sections are meshed with curved Quad8 shells, and again with Quad4. Flat shells
(Quad4, flat Quad8) with skins left on the outer mould line and offset lose
torsional stiffness in thick-walled closed sections, which is why the mid-surface is
the default; the last figure measures that on a thick tube and on the blade.

Figures: the six sections with their centres; the properties along the span against
PreComp and the published data; and the torsion check.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.iea15mw import DATA, build_segment, material_cards
from _shared.plotting import GRID, REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import (
    Airfoil,
    BladeBuilder,
    FSDTModel,
    Laminate,
    Ply,
    SkinRegion,
    Station,
    beam_section_properties,
)

LENGTH_CHORDS = 8.0   # segment length in chords (the middle fifth is measured)
ELEMENT = "quad8_curved"
N_SPAN = 32           # elements along the segment
H_PERIMETER = 0.015   # chordwise element size as a fraction of the section perimeter
QUAD4 = {"element": "quad4", "n_span": 40, "h_perimeter": 0.01}  # the flat-shell comparison
TUBE_T_OVER_R = [0.005, 0.01, 0.02, 0.04, 0.06]

# Section drawing: one colour per material family
MATERIAL_COLOURS = {"CarbonUD": SERIES[0], "glass_uni": SERIES[1], "medium_density_foam": SERIES[3],
                    "glass_triax": SERIES[2], "glass_biax": SERIES[2], "Gelcoat": GRID}


def build_station(st, length_chords=LENGTH_CHORDS, n_span=N_SPAN, h_perimeter=H_PERIMETER, element=ELEMENT):
    """A prismatic shell segment of one station, ``length_chords`` chords long."""
    return build_segment(st, length_chords * st["chord"], n_span, h_perimeter, element)


# ---------------------------------------------------------------------------
# Comparable quantities from each source
# ---------------------------------------------------------------------------
def summary_from_stiffness(S4, chord, pitch):
    """EA, bending about the tension centre (chord frame and principal), GJ and the
    tension centre from a 4 x 4 [N, Mx, My, Mz] stiffness about the pitch axis."""
    C = np.linalg.inv(S4)
    EA = S4[0, 0]
    B = S4[1:3, 1:3] - np.outer(S4[1:3, 0], S4[0, 1:3]) / EA  # bending about the tension centre
    # tension centre: an axial force there causes no curvature
    xt, yt = np.linalg.solve(np.column_stack([-C[1:3, 2], C[1:3, 1]]), -C[1:3, 0])
    return {"EA": EA, "EI_flap": B[0, 0], "EI_edge": B[1, 1], "EI_flap_edge": B[0, 1],
            "EI_min": np.linalg.eigvalsh(B)[0], "EI_max": np.linalg.eigvalsh(B)[1], "GJ": 1.0 / C[3, 3],
            "tc": ((pitch * chord + xt) / chord, yt / chord)}


def shell_summary(props, st):
    out = summary_from_stiffness(props.stiffness, st["chord"], st["pitch_axis"])
    c, pa = st["chord"], st["pitch_axis"]
    out["sc"] = ((pa * c + props.shear_centre[0]) / c, props.shear_centre[1] / c)
    out["cg"] = ((pa * c + props.mass_centre[0]) / c, props.mass_centre[1] / c)
    out["tc"] = ((pa * c + props.tension_centre[0]) / c, props.tension_centre[1] / c)
    out["mass"] = props.mass_per_length
    return out


def precomp_summary(st):
    """PreComp axes: x flapwise (suction side positive), y chordwise (aft positive),
    both from the pitch axis; bending is about the tension centre in the chord frame."""
    p, c, pa = st["precomp"], st["chord"], st["pitch_axis"]
    B = np.array([[p["EI_flap"], p["EI_flap_edge"]], [p["EI_flap_edge"], p["EI_edge"]]])
    lam = np.linalg.eigvalsh(B)
    centre = lambda key: ((pa * c + p[f"y_{key}"]) / c, p[f"x_{key}"] / c)  # noqa: E731
    return {"EA": p["EA"], "EI_flap": p["EI_flap"], "EI_edge": p["EI_edge"], "EI_flap_edge": p["EI_flap_edge"],
            "EI_min": lam[0], "EI_max": lam[1], "GJ": p["GJ"], "mass": p["mass"],
            "tc": centre("tc"), "sc": centre("sc"), "cg": centre("cg")}


def published_summary(st):
    """The published 6 x 6 (BeamDyn order: shear x, shear y, axial, bending x, bending y,
    torsion). Stiffnesses use only frame-free quantities: EA, the principal bending
    stiffnesses about the tension centre and GJ, all with the shear forces free (the
    4 x 4 block of the compliance). Its axes turn out to be chord-aligned (x flapwise,
    y chordwise aft, from the pitch axis: its tension centre matches PreComp's), so the
    centres are given too."""
    C = np.linalg.inv(np.asarray(st["published"]["K"]))
    C4 = C[2:, 2:]
    S4 = np.linalg.inv(C4)
    B = S4[1:3, 1:3] - np.outer(S4[1:3, 0], S4[0, 1:3]) / S4[0, 0]
    lam = np.linalg.eigvalsh(B)
    c, pa = st["chord"], st["pitch_axis"]
    # tension centre: an axial force there causes no curvature
    a_t, b_t = np.linalg.solve(np.column_stack([-C4[1:3, 2], C4[1:3, 1]]), -C4[1:3, 0])
    # shear centre: a shear force there causes no twist (torque a Fy - b Fx about the reference)
    a_s, b_s = -C[5, 1] / C[5, 5], C[5, 0] / C[5, 5]
    return {"EA": S4[0, 0], "EI_min": lam[0], "EI_max": lam[1], "GJ": 1.0 / C4[3, 3], "mass": st["published"]["mass"],
            "tc": (pa + b_t / c, a_t / c), "sc": (pa + b_s / c, a_s / c)}


def thick_tube_gj(t_over_r, element, mesh_surface, radius=1.0):
    """GJ of a thick isotropic tube (outer radius ``radius``) over the exact value G 2 pi R^3 t
    (R the mid-wall radius), for a shell model meshed on the outer surface (skin offset
    inward) or on the skin's mid-surface."""
    steel = {"E1": 2.0e11, "E2": 2.0e11, "G12": 2.0e11 / 2.6, "nu12": 0.3, "rho": 7850.0}
    card = material_cards({"steel": steel})["steel"]
    t = t_over_r * radius
    tube = Airfoil.ellipse(1.0)
    stations = [Station(0.0, 2 * radius, tube, pitch_axis=0.5), Station(8 * radius, 2 * radius, tube, pitch_axis=0.5)]
    n = 12 if element != "quad4" else 24
    blade = BladeBuilder(stations, [SkinRegion("upper", 0.0, 0.5, n), SkinRegion("lower", 0.5, 1.0, n)], n_span=16,
                         element=element, mesh_surface=mesh_surface).build(Laminate([Ply(card, t, 0.0)]), FSDTModel())
    props = beam_section_properties(blade)
    r_mid = radius - 0.5 * t
    return (1.0 / np.linalg.inv(props.stiffness)[3, 3]) / (steel["G12"] * 2 * np.pi * r_mid**3 * t)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def draw_section(ax, blade, st, shell, pc, pub):
    mesh, c, pa = blade.mesh, st["chord"], st["pitch_axis"]
    z_cut = blade.span_positions[len(blade.span_positions) // 2]
    for eid in blade.elements("all"):
        e = mesh.element(eid)
        xy = np.array([mesh.node_coordinates(n) for n in e.node_ids[:4]])
        xy = xy[np.abs(xy[:, 2] - z_cut) < 1e-9]
        if len(xy) != 2:
            continue
        plies = e.laminate.plies
        main = max(plies, key=lambda p: p.thickness)
        h = e.laminate.thickness
        colour = MATERIAL_COLOURS.get(main.material.name, TEXT_SECONDARY)
        # the laminate's mid-surface, drawn about as thick as the laminate
        mid = xy[:, :2] + e.laminate.mid_plane * np.asarray(e.frame)[2][:2]
        pts = (mid + [pa * c, 0.0]) / c
        ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=max(0.5, 330.0 * h / c), solid_capstyle="butt")
    for label, key, marker, colour in (("tension centre", "tc", "s", SERIES[4]), ("shear centre", "sc", "o", SERIES[6]),
                                       ("mass centre", "cg", "^", SERIES[5])):
        ax.plot(*shell[key], marker, color=colour, ms=7, mfc=colour, mew=1.2, label=f"{label}, shell")
        ax.plot(*pc[key], marker, color=colour, ms=11, mfc="none", mew=1.2, label=f"{label}, PreComp")
    ax.plot(*pub["sc"], "x", color=SERIES[6], ms=8, mew=1.6, label="shear centre, published")
    ax.plot(pa, 0.0, "+", color=REFERENCE, ms=9, mew=1.5, label="pitch axis")
    ax.set_aspect("equal")
    ax.set_title(f"r = {st['r']:.1f} m (span {st['span_fraction']:.2f}), chord {c:.2f} m", fontsize=9.5)
    ax.set_xlim(-0.05, 1.05)
    ax.tick_params(labelsize=7.5)


def figure_sections(models):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.5))
    for ax, (blade, st, shell, pc, pub) in zip(axes.ravel(), models):
        draw_section(ax, blade, st, shell, pc, pub)
    for ax in axes[-1]:
        ax.set_xlabel("x / chord from the leading edge")
    for ax in axes[:, 0]:
        ax.set_ylabel("y / chord")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for name, colour in (("carbon UD (spar caps)", SERIES[0]), ("glass UD (TE reinforcement)", SERIES[1]),
                         ("foam core", SERIES[3]), ("glass triax / biax", SERIES[2])):
        handles.append(plt.Line2D([], [], color=colour, lw=3))
        labels.append(name)
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("IEA 15 MW blade sections: centres from the shell model (filled), PreComp (open) and the published data (x)",
                 x=0.0, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, __file__, "iea15mw_sections.png")


def figure_properties(rows):
    s = np.array([r["st"]["span_fraction"] for r in rows])
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    panels = [("EA", "EA [N]"), ("EI_min", "flapwise EI (min principal) [N m²]"),
              ("EI_max", "edgewise EI (max principal) [N m²]"), ("GJ", "GJ [N m²]"), ("mass", "mass [kg/m]")]
    for ax, (key, label) in zip(axes.ravel(), panels):
        ax.plot(s, [r["shell"][key] for r in rows], "o-", color=SERIES[0], label="shell model (this package)")
        ax.plot(s, [r["pc"][key] for r in rows], "s--", color=SERIES[1], mfc="none", label="PreComp, same sections")
        ax.plot(s, [r["pub"][key] for r in rows], ":", marker="D", color=REFERENCE, mfc="none", ms=5,
                label="published IEA 15 MW data")
        ax.set_yscale("log")
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("span fraction")
    axes[0, 0].legend(fontsize=8)
    ax = axes[1, 2]
    for key, label, colour, marker in (("EA", "EA", SERIES[0], "o"), ("EI_min", "flap EI", SERIES[1], "s"),
                                       ("EI_max", "edge EI", SERIES[2], "^"), ("mass", "mass", SERIES[4], "v")):
        ax.plot(s, [r["shell"][key] / r["pc"][key] for r in rows], marker + "-", color=colour, label=label, ms=5)
    ax.axhline(1.0, color=REFERENCE, lw=0.8)
    ax.set_ylim(0.96, 1.04)
    ax.set_title("shell model / PreComp (GJ: torsion figure)", loc="left", fontsize=10)
    ax.set_xlabel("span fraction")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "iea15mw_properties.png")


def figure_torsion(tube, rows):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.2))
    t = np.array(TUBE_T_OVER_R)
    for (element, surface), colour, marker, label in (
            (("quad4", "mid"), SERIES[2], "^", "Quad4 on the mid-surface (default)"),
            (("quad4", "outer"), SERIES[1], "s", "Quad4 on the outer surface, skin offset"),
            (("quad8", "outer"), SERIES[3], "D", "flat Quad8, outer, offset"),
            (("quad8_curved", "outer"), SERIES[0], "o", "curved Quad8, outer, offset")):
        ax0.plot(t, tube[(element, surface)], marker + "-", color=colour, label=label)
    ax0.axhline(1.0, color=REFERENCE, lw=0.8)
    ax0.set_xlabel("wall thickness / radius")
    ax0.set_ylabel("GJ / exact")
    ax0.set_title("Thick steel tube: offset skins on flat shells lose torsion", loc="left")
    ax0.legend(fontsize=8)

    s = np.array([r["st"]["span_fraction"] for r in rows])
    ax1.plot(s, [r["shell"]["GJ"] / r["pc"]["GJ"] for r in rows], "o-", color=SERIES[0], label="curved Quad8 shell model")
    ax1.plot(s, [r["gj_quad4"] / r["pc"]["GJ"] for r in rows], "s-", color=SERIES[1], label="Quad4 shell model")
    ax1.plot(s, [r["pub"]["GJ"] / r["pc"]["GJ"] for r in rows], ":", marker="D", color=REFERENCE, mfc="none", ms=5,
             label="published IEA 15 MW data")
    ax1.axhline(1.0, color=TEXT_SECONDARY, lw=0.8)
    ax1.set_xlabel("span fraction")
    ax1.set_ylabel("GJ / PreComp")
    ax1.set_title("IEA 15 MW blade: GJ against PreComp", loc="left")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "iea15mw_torsion.png")


# ---------------------------------------------------------------------------
def print_station(st, shell, pc, pub, n_elements, seconds):
    print(f"\nr = {st['r']:5.1f} m (span {st['span_fraction']:.3f}), chord {st['chord']:.2f} m, "
          f"{n_elements} elements, {seconds:.0f} s")
    print(f"  {'':12}{'shell':>11}{'PreComp':>11}{'ratio':>8}{'published':>11}")
    for key, label in (("EA", "EA"), ("EI_flap", "EI flap"), ("EI_edge", "EI edge"), ("EI_min", "EI min"),
                       ("EI_max", "EI max"), ("GJ", "GJ"), ("mass", "mass")):
        published = f"{pub[key]:11.4g}" if key in pub else ""
        print(f"  {label:<12}{shell[key]:11.4g}{pc[key]:11.4g}{shell[key] / pc[key]:8.3f}{published}")
    for key, label in (("tc", "tension ctr"), ("sc", "shear ctr"), ("cg", "mass ctr")):
        published = f"   published {pub[key][0]:6.3f}" if key in pub else ""
        print(f"  {label:<12}x/c {shell[key][0]:6.3f} vs {pc[key][0]:6.3f}   "
              f"y/c {shell[key][1]:+6.3f} vs {pc[key][1]:+6.3f}{published}")


def main():
    print(f"{DATA['turbine']}: {len(DATA['stations'])} stations from {DATA['source']}")
    rows, models = [], []
    for st in DATA["stations"]:
        t0 = time.perf_counter()
        blade = build_station(st)
        shell = shell_summary(beam_section_properties(blade), st)
        pc, pub = precomp_summary(st), published_summary(st)
        print_station(st, shell, pc, pub, len(blade.mesh.element_ids), time.perf_counter() - t0)
        flat = shell_summary(beam_section_properties(build_station(st, **QUAD4)), st)
        print(f"  Quad4 model: GJ {flat['GJ']:.4g} ({flat['GJ'] / pc['GJ']:.3f} of PreComp), "
              f"EA {flat['EA'] / pc['EA']:.3f}, EI flap {flat['EI_flap'] / pc['EI_flap']:.3f}, "
              f"EI edge {flat['EI_edge'] / pc['EI_edge']:.3f}")
        rows.append({"st": st, "shell": shell, "pc": pc, "pub": pub, "gj_quad4": flat["GJ"]})
        models.append((blade, st, shell, pc, pub))

    print("\nThick tube, GJ / exact:")
    tube = {}
    for key in (("quad4", "mid"), ("quad4", "outer"), ("quad8", "outer"), ("quad8_curved", "outer")):
        tube[key] = [thick_tube_gj(t, *key) for t in TUBE_T_OVER_R]
        print(f"  {key[0]:<13}{key[1]:<6}" + "".join(f"{v:8.3f}" for v in tube[key]))

    figure_sections(models)
    figure_properties(rows)
    figure_torsion(tube, rows)


if __name__ == "__main__":
    main()
