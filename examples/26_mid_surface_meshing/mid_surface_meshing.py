"""Blade skins meshed on the mould line with offset laminates, or on the mid-surface.

``BladeBuilder`` meshes skins on the outer mould line (OML) by default and offsets
each laminate inward by half its thickness. Example 24 found that flat facets
(Quad4, flat Quad8) then lose torsional stiffness in thick closed sections.
``mesh_surface="mid"`` instead moves the skin nodes onto the laminates' mid-surface,
smoothed across thickness changes, re-attaches the webs between the moved skins, and
drops the offsets. This example compares the two on:

1. a thick tube (closed-form GJ and EA), t/R from 0.01 to 0.06;
2. six stations of the IEA 15 MW blade against PreComp and the published data
   (the sections of example 24),

each with Quad4, flat Quad8 and curved Quad8.

The runs take about an hour and a half; they are saved to ``data/results.json``,
and later runs draw the figures from that file. ``--rerun`` repeats the analyses.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

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

# example 24's comparable quantities (PreComp, published data, shell model)
_spec = importlib.util.spec_from_file_location(
    "ex24", HERE.parent / "24_iea15mw_section_properties" / "iea15mw_section_properties.py")
ex24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex24)

RESULTS = HERE / "data" / "results.json"
ELEMENTS = {  # element: (elements along the segment, chordwise size / perimeter), as in example 24
    "quad4": (40, 0.010),
    "quad8": (32, 0.015),
    "quad8_curved": (32, 0.015),
}
SURFACES = ("outer", "mid")
LENGTH_CHORDS = 8.0
TUBE_T_OVER_R = [0.01, 0.02, 0.04, 0.06]
QUANTITIES = ("EA", "EI_min", "EI_max", "GJ", "mass")

ELEMENT_LABEL = {"quad4": "Quad4", "quad8": "flat Quad8", "quad8_curved": "curved Quad8"}
ELEMENT_COLOUR = {"quad4": SERIES[0], "quad8": SERIES[1], "quad8_curved": SERIES[2]}
SURFACE_STYLE = {"outer": dict(ls="--", marker="o", mfc="none"), "mid": dict(ls="-", marker="o")}
SURFACE_LABEL = {"outer": "mould line + offset", "mid": "mid-surface"}


# ---------------------------------------------------------------------------
# 1. Thick tube
# ---------------------------------------------------------------------------
def tube(t_over_r, element, surface, radius=1.0):
    """GJ and EA of a steel tube (outer radius ``radius``, wall t inward) over the exact values."""
    steel = {"E1": 2.0e11, "E2": 2.0e11, "G12": 2.0e11 / 2.6, "nu12": 0.3, "rho": 7850.0}
    card = material_cards({"steel": steel})["steel"]
    t = t_over_r * radius
    circle = Airfoil.ellipse(1.0)
    stations = [Station(0.0, 2 * radius, circle, pitch_axis=0.5), Station(8 * radius, 2 * radius, circle, pitch_axis=0.5)]
    n = 24 if element == "quad4" else 12
    blade = BladeBuilder(stations, [SkinRegion("upper", 0.0, 0.5, n), SkinRegion("lower", 0.5, 1.0, n)], n_span=16,
                         element=element, mesh_surface=surface).build(Laminate([Ply(card, t, 0.0)]), FSDTModel())
    props = beam_section_properties(blade)
    r_o, r_i = radius, radius - t
    J = 0.5 * np.pi * (r_o**4 - r_i**4)
    A = np.pi * (r_o**2 - r_i**2)
    return {"GJ": (1.0 / np.linalg.inv(props.stiffness)[3, 3]) / (steel["G12"] * J),
            "EA": props.EA / (steel["E1"] * A)}


# ---------------------------------------------------------------------------
# 2. IEA 15 MW stations
# ---------------------------------------------------------------------------
def station(st, element, surface):
    n_span, h = ELEMENTS[element]
    t0 = time.time()
    blade = build_segment(st, LENGTH_CHORDS * st["chord"], n_span, h, element, mesh_surface=surface)
    s = ex24.shell_summary(beam_section_properties(blade), st)
    out = {q: float(s[q]) for q in QUANTITIES}
    out["sc"] = [float(v) for v in s["sc"]]
    out["seconds"] = time.time() - t0
    return out


def run():
    results = {"tube": {}, "stations": []}
    for element in ELEMENTS:
        for surface in SURFACES:
            key = f"{element}/{surface}"
            results["tube"][key] = [tube(r, element, surface) for r in TUBE_T_OVER_R]
            print(f"tube {key:24s} GJ/exact " + " ".join(f"{v['GJ']:.3f}" for v in results["tube"][key]), flush=True)
    for st in DATA["stations"]:
        pc, pub = ex24.precomp_summary(st), ex24.published_summary(st)
        row = {"r": st["r"], "chord": st["chord"],
               "precomp": {q: float(pc[q]) for q in QUANTITIES},
               "published": {q: float(pub[q]) for q in QUANTITIES},
               "precomp_sc": [float(v) for v in pc["sc"]], "published_sc": [float(v) for v in pub["sc"]],
               "shell": {}}
        for element in ELEMENTS:
            for surface in SURFACES:
                s = station(st, element, surface)
                row["shell"][f"{element}/{surface}"] = s
                print(f"r = {st['r']:5.1f} m {element:13s} {surface:5s} "
                      + " ".join(f"{q} {s[q] / pc[q]:.3f}" for q in QUANTITIES)
                      + f"  (/published GJ {s['GJ'] / pub['GJ']:.3f}, EA {s['EA'] / pub['EA']:.3f})"
                      + f"  sc x/c {s['sc'][0]:.3f}  {s['seconds']:.0f} s", flush=True)
        results["stations"].append(row)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_tube(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, q, title in ((axes[0], "GJ", "Torsion: GJ / exact"), (axes[1], "EA", "Axial: EA / exact")):
        for element in ELEMENTS:
            for surface in SURFACES:
                v = [r[q] for r in results["tube"][f"{element}/{surface}"]]
                ax.plot(TUBE_T_OVER_R, v, color=ELEMENT_COLOUR[element], **SURFACE_STYLE[surface],
                        label=f"{ELEMENT_LABEL[element]}, {SURFACE_LABEL[surface]}")
        ax.axhline(1.0, color=REFERENCE, lw=0.8)
        ax.set_xlabel("wall thickness / outer radius t/R")
        ax.set_title(title)
    axes[0].set_ylabel("shell model / exact thick tube")
    axes[1].set_ylim(0.96, 1.04)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.text(0.0, -0.04, "Steel tube, outer radius 1 m, skin on the outer surface growing inward; 16 elements along "
             "8 m, 24 (Quad4) or 12 (Quad8) around each half. Exact: J = pi/2 (Ro^4 - Ri^4), A = pi (Ro^2 - Ri^2).",
             fontsize=8, color=TEXT_SECONDARY)
    save(fig, __file__, "mid_surface_tube.png")


def figure_blade(results):
    rows = results["stations"]
    x = [row["r"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), gridspec_kw={"hspace": 0.38})
    panels = [("GJ", "precomp", "GJ / PreComp"), ("EA", "precomp", "EA / PreComp"), ("mass", "precomp", "mass / PreComp"),
              ("GJ", "published", "GJ / published"), ("EA", "published", "EA / published"),
              ("mass", "published", "mass / published")]
    for ax, (q, ref, title) in zip(axes.ravel(), panels):
        for element in ELEMENTS:
            for surface in SURFACES:
                v = [row["shell"][f"{element}/{surface}"][q] / row[ref][q] for row in rows]
                ax.plot(x, v, color=ELEMENT_COLOUR[element], **SURFACE_STYLE[surface],
                        label=f"{ELEMENT_LABEL[element]}, {SURFACE_LABEL[surface]}")
        ax.axhline(1.0, color=REFERENCE, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("blade radius r [m]")
        if q == "GJ" and ref == "precomp":
            ax.set_yscale("log")
            ax.set_yticks([0.75, 1.0, 1.5, 2.0])
            ax.set_yticklabels(["0.75", "1", "1.5", "2"])
        elif q != "GJ":
            ax.set_ylim(0.96, 1.02) if ref == "precomp" else ax.set_ylim(0.94, 1.12)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.text(0.0, -0.02, "IEA 15 MW blade, the six sections of example 24, each a prismatic segment 8 chords long. "
             "Dashed: skins on the outer mould line with laminates offset inward. Solid: skins on the laminates' "
             "mid-surface.", fontsize=8, color=TEXT_SECONDARY)
    save(fig, __file__, "mid_surface_iea15mw.png")


def figure_section():
    """One section on both surfaces, with close-ups of a cap edge and the trailing edge."""
    st = DATA["stations"][1]
    c, pa = st["chord"], st["pitch_axis"]
    blades = {s: build_segment(st, LENGTH_CHORDS * c, 2, 0.015, "quad4", mesh_surface=s) for s in SURFACES}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    for surface, blade in blades.items():
        mesh = blade.mesh
        z = blade.span_positions[1]
        for eid in blade.elements("all"):
            e = mesh.element(eid)
            xy = np.array([mesh.node_coordinates(n) for n in e.node_ids[:4]])
            xy = xy[np.abs(xy[:, 2] - z) < 1e-9][:, :2]
            if len(xy) != 2:
                continue
            pts = (xy + [pa * c, 0.0]) / c
            for ax in axes:
                ax.plot(pts[:, 0], pts[:, 1], color=SERIES[0] if surface == "mid" else GRID if ax is axes[0] else
                        TEXT_SECONDARY, lw=1.4 if surface == "mid" else 1.0, ls="-" if surface == "mid" else "--")
    axes[0].plot([], [], color=TEXT_SECONDARY, ls="--", label="outer mould line (default mesh)")
    axes[0].plot([], [], color=SERIES[0], label="mid-surface mesh")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"r = {st['r']:.1f} m, chord {c:.2f} m")
    zooms = [(axes[1], (0.12, 0.30), "spar-cap edge, suction side"), (axes[2], (0.86, 1.01), "trailing edge (flatback)")]
    for ax, (x0, x1), title in zooms:
        ax.set_xlim(x0, x1)
        ys = []
        for blade in blades.values():
            for n in blade.section_nodes(blade.span_positions[1]):
                p = (blade.mesh.node_coordinates(n)[:2] + [pa * c, 0.0]) / c
                if x0 <= p[0] <= x1 and (p[1] > 0 or "trailing" in title):
                    ys.append(p[1])
        lo, hi = min(ys), max(ys)
        pad = 0.15 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title)
        ax.set_xlabel("x / chord")
    axes[0].set_xlabel("x / chord")
    axes[0].set_ylabel("y / chord")
    fig.text(0.0, -0.04, "Skin nodes move inward by half the local laminate thickness, averaged over the elements "
             "at each node and smoothed along the skin over two of the thickest laminates; webs end on the moved "
             "skins. Quad4 mesh shown.", fontsize=8, color=TEXT_SECONDARY)
    save(fig, __file__, "mid_surface_section.png")


def print_table(results):
    print("\nGJ / PreComp and EA, mass / PreComp, curved Quad8 and Quad4 on both surfaces")
    for row in results["stations"]:
        cells = []
        for key in ("quad8_curved/outer", "quad8_curved/mid", "quad4/outer", "quad4/mid"):
            s = row["shell"][key]
            cells.append(f"{key:19s} GJ {s['GJ'] / row['precomp']['GJ']:.3f} EA {s['EA'] / row['precomp']['EA']:.3f}")
        print(f"r = {row['r']:5.1f}  " + " | ".join(cells))


def main():
    if RESULTS.exists() and "--rerun" not in sys.argv:
        results = json.loads(RESULTS.read_text())
    else:
        results = run()
    print_table(results)
    figure_tube(results)
    figure_blade(results)
    figure_section()


if __name__ == "__main__":
    main()
