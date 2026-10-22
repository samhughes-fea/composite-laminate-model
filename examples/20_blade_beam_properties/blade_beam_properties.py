"""Beam properties of a blade section from its shell model, and bend-twist tailoring.

A prismatic blade segment (NACA 0024, 1 m chord, 6 m long) has glass skins, a
carbon spar cap on each surface and two shear webs. ``beam_section_properties``
loads the shell model as a cantilever and returns the section's 4 x 4 stiffness
(axial, two bending, torsion, with all couplings), shear and tension centres and
mass per length, the inputs a beam or aeroelastic model needs.

The caps are then tailored: their fibres are turned by +theta on the suction side
and -theta on the pressure side (a circumferentially asymmetric layup). That couples
flap bending to twist, the mechanism behind bend-twist-coupled, load-alleviating
blades, at the cost of flap stiffness.

Left: the section, with the centres. Middle: flap and torsional stiffness against
cap angle. Right: the bend-twist coupling coefficient.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import (
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

CHORD, LENGTH, N_SPAN = 1.0, 6.0, 20
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

    sweep = []
    for angle in ANGLES:
        p = beam_section_properties(build(laminates(glass, carbon, angle)))
        sweep.append(p)
        print(f"cap angle {angle:3d} deg: EI flap {p.EI_x / base.EI_x:.3f}, GJ {p.GJ / base.GJ:.3f}, "
              f"bend-twist coupling {p.bend_twist_coupling[0]:+.3f}")

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
    ax1.set_xlabel("cap fibre angle θ [deg]")
    ax1.set_ylabel("stiffness / stiffness at θ = 0")
    ax1.set_title("Tailoring costs flap stiffness", loc="left")
    ax1.legend()

    ax2.plot(a, [p.bend_twist_coupling[0] for p in sweep], "o-", color=SERIES[3])
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
