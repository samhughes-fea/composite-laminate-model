"""Beam properties of a blade section from its shell model, and bend-twist tailoring.

A prismatic blade segment (NACA 0024, 1 m chord, 12 m long) has glass skins, a
carbon spar cap on each surface and two shear webs. ``beam_section_properties``
loads the shell model as a cantilever and returns the section's 4 x 4 stiffness
(axial, two bending, torsion, with all couplings), shear and tension centres and
mass per length, the inputs a beam or aeroelastic model needs.

The caps are then tailored: their fibres are turned by +theta on the suction side
and -theta on the pressure side (a circumferentially asymmetric layup). That couples
flap bending to twist, the mechanism behind bend-twist-coupled, load-alleviating
blades, at the cost of flap stiffness.

Two references are computed for every cap angle:

* thin-walled beam theory: each wall is a laminate (its A matrix, hoop stress free),
  the shear flows in the three cells follow from Bredt-Batho compatibility, and the
  section stiffness from the wall forces. It is exact in the thin-wall limit for a
  long, prismatic segment with free warping.
* PreComp (NREL's thin-wall section code, WISDEM's Python port), run on the same
  section; its output is in ``data/precomp_reference.json``.

Left: the section, with the centres. Middle: flap and torsional stiffness against
cap angle. Right: the bend-twist coupling coefficient.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.thin_wall import thin_wall_stiffness
from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from composite_laminate_model import (
    Airfoil,
    BladeBuilder,
    FSDTModel,
    Laminate,
    ShearWeb,
    SkinRegion,
    Station,
    beam_section_properties,
    create_default_library,
)

# 12 chords: the clamp's and the end rib's restraint of warping must die out before the
# middle fifth, where the properties are measured. With off-axis caps that takes long:
# a 6 m segment gave EI flap 9% and the bend-twist coupling 23% too high at 20 degrees.
CHORD, LENGTH, N_SPAN = 1.0, 12.0, 40
REGIONS = [SkinRegion("le_panel", 0.0, 0.18, 4), SkinRegion("spar_cap", 0.18, 0.48, 6),
           SkinRegion("te_panel", 0.48, 1.0, 8)]
WEBS = [ShearWeb("web_fwd", 0.18, 4), ShearWeb("web_aft", 0.48, 4)]
ANGLES = [0, 5, 10, 15, 20, 25, 30]


def laminates(glass, carbon, cap_angle=0.0):
    skin = Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3)
    web = Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3)
    cap = lambda a: Laminate.from_angles(carbon, [a] * 16, 0.5e-3)  # noqa: E731
    return {"le_panel": skin, "te_panel": skin, "web_fwd": web, "web_aft": web,
            "spar_cap_ss": cap(cap_angle), "spar_cap_ps": cap(-cap_angle)}


def thin_wall(lams):
    """Thin-walled beam theory for this section (``_shared/thin_wall.py``)."""
    return thin_wall_stiffness(Airfoil.naca4("0024"), CHORD, 0.3, REGIONS, WEBS, lams)


def coupling(S):
    """Flap-twist coupling S[1,3] / sqrt(S11 S33)."""
    return S[1, 3] / np.sqrt(S[1, 1] * S[3, 3])


def edge_about_tension_centre(S):
    """Edgewise bending stiffness about the tension centre."""
    return S[2, 2] - S[0, 2] ** 2 / S[0, 0]


def build(lams):
    airfoil = Airfoil.naca4("0024")
    stations = [Station(0.0, CHORD, airfoil, pitch_axis=0.3), Station(LENGTH, CHORD, airfoil, pitch_axis=0.3)]
    return BladeBuilder(stations, REGIONS, WEBS, n_span=N_SPAN).build(lams, FSDTModel())


def main():
    lib = create_default_library()
    glass, carbon = lib.get("E-glass/epoxy"), lib.get("T300/5208")
    blade = build(laminates(glass, carbon))
    base = beam_section_properties(blade)
    S = base.stiffness
    print("Section stiffness about the pitch axis [N, N m, N m^2]:")
    for row, name in zip(S, ("N", "Mx", "My", "Mz")):
        print(f"  {name:<3}" + "".join(f"{v:12.4g}" for v in row))
    print(f"EA {base.EA:.4g} N, EI flap {base.EI_x:.4g} N m2, EI edge {base.EI_y:.4g} N m2, GJ {base.GJ:.4g} N m2")
    print(f"shear centre {np.round(base.shear_centre, 4)} m, tension centre {np.round(base.tension_centre, 4)} m, "
          f"mass {base.mass_per_length:.2f} kg/m at {np.round(base.mass_centre, 4)} m")

    precomp = {r["angle"]: r for r in json.loads((Path(__file__).parent / "data" / "precomp_reference.json")
                                                 .read_text())["sweep"]}
    sweep, thin = [], []
    print("cap   EI flap [N m2]                   GJ [N m2]                        flap-twist coupling")
    print("[deg] shell     thin wall  PreComp    shell     thin wall  PreComp    shell   thin wall  PreComp")
    for angle in ANGLES:
        p = beam_section_properties(build(laminates(glass, carbon, angle))) if angle else base
        sweep.append(p)
        tw = thin_wall(laminates(glass, carbon, angle))
        thin.append(tw)
        pc = precomp[angle]
        print(f"{angle:4d}  {p.EI_x:.3e} {tw[1, 1]:.3e}  {pc['EI_flap']:.3e}  {p.GJ:.3e} {tw[3, 3]:.3e}  "
              f"{pc['GJ']:.3e}  {p.bend_twist_coupling[0]:+.3f}  {coupling(tw):+.3f}     "
              f"{pc['flap_GJ'] / np.sqrt(pc['EI_flap'] * pc['GJ']):+.3f}")
    tw0, pc0 = thin[0], precomp[0]
    print(f"cap angle 0: EA shell {base.EA:.4g}, thin wall {tw0[0, 0]:.4g}, PreComp {pc0['EA']:.4g}; "
          f"EI edge about the tension centre shell {edge_about_tension_centre(base.stiffness):.4g}, "
          f"thin wall {edge_about_tension_centre(tw0):.4g}, PreComp {pc0['EI_edge']:.4g}")
    print(f"tension centre x: shell {base.tension_centre[0]:.4f} m, thin wall {tw0[0, 2] / -tw0[0, 0]:.4f} m, "
          f"PreComp {pc0['y_tc']:.4f} m; shear centre x: shell {base.shear_centre[0]:.4f} m, PreComp {pc0['y_sc']:.4f} m")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1.0, 1.0]})
    mesh = blade.mesh
    colours = {"le_panel": SERIES[0], "spar_cap": SERIES[1], "te_panel": SERIES[0], "web_fwd": SERIES[2],
               "web_aft": SERIES[2]}
    z_cut = blade.span_positions[N_SPAN // 2]
    for name, colour in colours.items():
        for eid in blade.elements(name):
            e = mesh.element(eid)
            xy = np.array([mesh.node_coordinates(n) for n in e.node_ids[:4]])
            xy = xy[np.abs(xy[:, 2] - z_cut) < 1e-9]  # the element edge on this section
            if len(xy) == 2:
                ax0.plot(xy[:, 0], xy[:, 1], color=colour, lw=3.0 if name == "spar_cap" else 1.6,
                         solid_capstyle="butt")
    marks = [("pitch axis", (0.0, 0.0), "+", REFERENCE), ("shear centre", base.shear_centre, "o", SERIES[3]),
             ("tension centre", base.tension_centre, "s", SERIES[4]), ("mass centre", base.mass_centre, "^", SERIES[5])]
    for label, (x, y), marker, colour in marks:
        ax0.plot(x, y, marker, color=colour, ms=8, mfc="none" if marker != "+" else colour, mew=1.8, label=label)
    ax0.plot([], [], color=SERIES[1], lw=3, label="carbon spar caps")
    ax0.plot([], [], color=SERIES[0], lw=1.6, label="glass skins")
    ax0.plot([], [], color=SERIES[2], lw=1.6, label="shear webs")
    ax0.set_aspect("equal")
    ax0.set_xlabel("x, chordwise [m]")
    ax0.set_ylabel("y [m]")
    ax0.set_title("Section and its centres", loc="left")
    ax0.legend(loc="upper right", ncol=2, fontsize=7.5)
    ax0.set_ylim(-0.2, 0.36)

    a = np.array(ANGLES)
    ax1.plot(a, [p.EI_x / base.EI_x for p in sweep], "o-", color=SERIES[0], label="EI flap")
    ax1.plot(a, [p.GJ / base.GJ for p in sweep], "s-", color=SERIES[1], label="GJ")
    ax1.plot(a, [p.EA / base.EA for p in sweep], "^-", color=SERIES[2], label="EA")
    ax1.plot(a, [t[1, 1] / base.EI_x for t in thin], color=REFERENCE, lw=1.0, ls=(0, (4, 2)), label="thin-wall theory")
    ax1.plot(a, [t[3, 3] / base.GJ for t in thin], color=REFERENCE, lw=1.0, ls=(0, (4, 2)))
    ax1.plot(a, [t[0, 0] / base.EA for t in thin], color=REFERENCE, lw=1.0, ls=(0, (4, 2)))
    ax1.plot(a, [precomp[g]["EI_flap"] / base.EI_x for g in ANGLES], "o", mfc="none", color=REFERENCE, ms=5,
             label="PreComp")
    ax1.plot(a, [precomp[g]["GJ"] / base.GJ for g in ANGLES], "s", mfc="none", color=REFERENCE, ms=5)
    ax1.set_xlabel("cap fibre angle θ [deg]")
    ax1.set_ylabel("stiffness / stiffness at θ = 0")
    ax1.set_title("Tailoring costs flap stiffness", loc="left")
    ax1.legend(fontsize=7.5)

    ax2.plot(a, [p.bend_twist_coupling[0] for p in sweep], "o-", color=SERIES[3], label="shell model")
    ax2.plot(a, [coupling(t) for t in thin], color=REFERENCE, lw=1.0, ls=(0, (4, 2)), label="thin-wall theory")
    ax2.plot(a, [precomp[g]["flap_GJ"] / np.sqrt(precomp[g]["EI_flap"] * precomp[g]["GJ"]) for g in ANGLES], "o",
             mfc="none", color=REFERENCE, ms=5, label="PreComp")
    ax2.legend(loc="center right", fontsize=7.5)
    best = int(np.argmax(np.abs([p.bend_twist_coupling[0] for p in sweep])))
    ax2.annotate(f"{sweep[best].bend_twist_coupling[0]:+.2f} at {ANGLES[best]}°", (a[best], sweep[best].bend_twist_coupling[0]),
                 textcoords="offset points", xytext=(-30, 12), fontsize=8, color=TEXT_SECONDARY)
    ax2.axhline(0.0, color=TEXT_SECONDARY, lw=0.8)
    ax2.set_xlabel("cap fibre angle θ [deg]")
    ax2.set_ylabel("S₁₄ / √(S₁₁ S₄₄)  (flap-twist)")
    ax2.set_title("Bend-twist coupling", loc="left")
    fig.text(0.0, -0.05, f"NACA 0024, chord {CHORD:g} m, {LENGTH:g} m segment, {N_SPAN} elements along the span, curved Quad8 FSDT. "
             "Skins [0/±45]s E-glass 3 mm; caps 16 x 0.5 mm T300/5208 at +θ (suction) / −θ (pressure), 18-48% chord; "
             "webs [±45]s E-glass at 18% and 48%.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "blade_beam_properties.png")


if __name__ == "__main__":
    main()
