"""Native solver against CalculiX on a 20 m segment of the IEA 15 MW blade.

The segment runs from r = 64.5 m to r = 84.7 m with the published layup of the
r = 64.5 m station (carbon spar caps, foam-cored panels, glass trailing-edge
reinforcement, two sandwich webs, a flatback trailing edge closed by a web),
tapering from 4.07 m to 3.27 m chord with the published twist. The root is
clamped. It carries an operating-type load:

* suction on the suction side and pressure on the pressure side, both rising 30%
  along the segment (a flapwise load of about 10 kN/m),
* the blade's own weight edgewise (the blade horizontal),
* the shear of the outboard blade at the tip: 250 kN flapwise on the web nodes and
  60 kN edgewise on the skin nodes, as nodal forces.

The same ``AnalysisModel`` goes through ``NativePythonKernelAdapter`` and
``CalculiXKernelAdapter``. CalculiX takes composite shells only as S8R, so the
Quad4 model is promoted to S8R with straight-sided mid-side nodes; the curved
Quad8 model is written node for node. CalculiX expands S8R into 3D bricks, so it
also checks the offset skins, the thick caps and the web-to-skin joints.

Figures: deflection and twist along the span; spar-cap fibre stress along the span;
ply stresses through a cap and a sandwich panel.

CalculiX must be on the PATH as ``ccx`` (``apt install calculix-ccx``); without it
the script prints a message and stops.
"""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.iea15mw import DATA, build_segment
from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save
from composite_laminate_model import (
    AnalysisModel,
    BoundaryConditions,
    CalculiXKernelAdapter,
    LoadCase,
    NativePythonKernelAdapter,
    StressDistributionVerifier,
)

ROOT, TIP = DATA["stations"][3], DATA["stations"][4]  # r = 64.5 m and r = 84.7 m
LENGTH = TIP["r"] - ROOT["r"]
MODELS = {"Quad4": {"element": "quad4", "n_span": 40, "h_perimeter": 0.012},
          "curved Quad8": {"element": "quad8_curved", "n_span": 20, "h_perimeter": 0.024}}
Q_SUCTION, Q_PRESSURE = 1.7e3, 0.8e3  # at the segment root [Pa]; both rise 30% to the tip
TIP_FLAP, TIP_EDGE = 250.0e3, -60.0e3  # outboard shear [N]: flapwise (+y), edgewise (towards the LE)
GRAVITY = (-9.81, 0.0, 0.0)
WORK_DIR = Path(__file__).resolve().parent / "calculix_run"


def build(options):
    blade = build_segment(ROOT, LENGTH, tip_chord=TIP["chord"], twist=(ROOT["twist_deg"], TIP["twist_deg"]),
                          **options)
    rise = lambda q: (lambda x: q * (1.0 + 0.3 * x[2] / LENGTH))  # noqa: E731
    load = LoadCase("operating").add_body_load(GRAVITY)
    upper = [e for name, ids in blade.element_sets.items() if name.endswith("_ss") for e in ids]
    lower = [e for name, ids in blade.element_sets.items() if name.endswith("_ps") for e in ids]
    load.add_pressure(rise(Q_SUCTION), upper)    # +normal is outward: suction pulls the suction side out
    load.add_pressure(rise(-Q_PRESSURE), lower)  # pressure pushes the pressure side in
    tip = set(blade.nodes("tip"))
    web_tip = sorted(tip & {n for w in ("web0", "web1") for n in blade.nodes(w)})
    for n in web_tip:
        load.add_nodal_load(n, 1, TIP_FLAP / len(web_tip))
    for n in sorted(tip):
        load.add_nodal_load(n, 0, TIP_EDGE / len(tip))
    bcs = BoundaryConditions().clamp(blade.nodes("root"))
    return blade, bcs, load


def cap_elements(blade):
    """Per span row, the suction- and pressure-side cap elements nearest the cap centre line."""
    mesh = blade.mesh
    rows = {"ss": {}, "ps": {}}
    for name, ids in blade.element_sets.items():
        side = name[-2:]
        if side not in rows or not name.startswith("sector"):
            continue
        for eid in ids:
            e = mesh.element(eid)
            if not any(p.material.name == "CarbonUD" for p in e.laminate.plies):
                continue
            c = np.asarray(e.centroid)
            rows[side].setdefault(round(c[2], 6), []).append((eid, c))
    out = {}
    for side, by_z in rows.items():
        picks = []
        for z, items in sorted(by_z.items()):
            xs = np.array([c[0] for _, c in items])
            picks.append(items[int(np.argmin(np.abs(xs - xs.mean())))][0])
        out[side] = picks
    return out


def panel_element(blade, z_target):
    """A suction-side trailing-edge sandwich (foam-cored) element near span z_target, mid-panel."""
    mesh = blade.mesh
    best, score = None, np.inf
    for name, ids in blade.element_sets.items():
        if not (name.startswith("sector") and name.endswith("_ss")):
            continue
        for eid in ids:
            e = mesh.element(eid)
            if not any(p.material.name == "medium_density_foam" for p in e.laminate.plies):
                continue
            c = np.asarray(e.centroid)
            s = abs(c[2] - z_target) + 0.5 * abs(c[0] - 0.6 * ROOT["chord"])  # aft panel, mid-way
            if s < score:
                best, score = eid, s
    return best


def section_motion(blade, displacements):
    """Mean edgewise and flapwise deflection and twist of each mesh section, from node translations."""
    mesh = blade.mesh
    out = []
    for z in blade.span_positions:
        nodes = [n for n in blade.section_nodes(float(z)) if n in displacements]
        X = np.array([mesh.node_coordinates(n)[:2] for n in nodes])
        U = np.array([displacements[n][:2] for n in nodes])
        r = X - X.mean(axis=0)
        u = U - U.mean(axis=0)
        twist = np.sum(r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0]) / np.sum(r**2)
        out.append((float(z), *U.mean(axis=0), twist))
    return np.array(out)


def ply_mid_value(profile, laminate, ply, component):
    """Mean of a component over the two Gauss heights of one ply."""
    k = 2 * ply
    return float(np.mean(profile.component(component)[k:k + 2]))


def run(label, options):
    blade, bcs, load = build(options)
    caps = cap_elements(blade)
    panel = panel_element(blade, 0.25 * LENGTH)
    cap_detail = caps["ss"][len(caps["ss"]) // 4]
    monitor = tuple(dict.fromkeys(caps["ss"] + caps["ps"] + [panel]))
    model = AnalysisModel(blade.mesh, bcs, load, title=f"iea15mw_{options['element']}", monitor_elements=monitor)
    print(f"\n{label}: {len(blade.mesh.element_ids)} elements, {len(blade.mesh.node_ids)} nodes")
    native = NativePythonKernelAdapter().run(model)
    if not native.ok:
        raise RuntimeError(native.message)
    ccx = CalculiXKernelAdapter(work_dir=WORK_DIR / options["element"], timeout=3600.0).run(model)
    if not ccx.ok:
        raise RuntimeError(f"CalculiX failed: {ccx.message}")
    print(f"  runtime: native {native.runtime:.1f} s, CalculiX {ccx.runtime:.1f} s "
          f"({ccx.info.get('n_nodes_s8r')} S8R nodes)")
    return {"blade": blade, "native": native, "ccx": ccx, "caps": caps, "panel": panel, "cap_detail": cap_detail}


def report(label, r):
    blade, native, ccx = r["blade"], r["native"], r["ccx"]
    verifier = StressDistributionVerifier()
    stats = verifier.compare_displacements(native.displacements, ccx.displacements)
    print(verifier.format_table(stats.values(), f"{label}: displacements, CalculiX vs native"))
    m_n, m_c = section_motion(blade, native.displacements), section_motion(blade, ccx.displacements)
    tip = -1
    print(f"  tip flapwise {m_n[tip, 2] * 1e3:.1f} mm native, {m_c[tip, 2] * 1e3:.1f} mm CalculiX "
          f"({m_n[tip, 2] / m_c[tip, 2]:.3f}); edgewise {m_n[tip, 1] * 1e3:.1f} vs {m_c[tip, 1] * 1e3:.1f} mm; "
          f"twist {np.degrees(m_n[tip, 3]):.3f} vs {np.degrees(m_c[tip, 3]):.3f} deg")
    for eid, name in ((r["cap_detail"], "suction cap"), (r["panel"], "sandwich panel")):
        ref, cand = native.profile(eid, "material"), ccx.profile(eid, "material")
        s = verifier.compare_profiles(ref, cand, components=("s11", "s22", "t12"))
        print(verifier.format_table(s.values(), f"{label}: ply stresses in the {name} (element {eid})"))
    r["motion"] = (m_n, m_c)
    return r


def cap_series(r, side):
    """Fibre stress at the middle of the carbon ply of each cap element along the span."""
    blade = r["blade"]
    z, sn, sc = [], [], []
    for eid in r["caps"][side]:
        e = blade.mesh.element(eid)
        ply = next(k for k, p in enumerate(e.laminate.plies) if p.material.name == "CarbonUD")
        z.append(float(np.asarray(e.centroid)[2]))
        sn.append(ply_mid_value(r["native"].profile(eid, "material"), e.laminate, ply, "s11"))
        sc.append(ply_mid_value(r["ccx"].profile(eid, "material"), e.laminate, ply, "s11"))
    return np.array(z), np.array(sn), np.array(sc)


def figure(results):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    (ax_d, ax_t), (ax_s, ax_p) = axes
    styles = {"Quad4": (SERIES[0], "o"), "curved Quad8": (SERIES[1], "s")}
    r_main = results["Quad4"]
    for label, r in results.items():
        colour, marker = styles[label]
        m_n, m_c = r["motion"]
        z = (m_n[:, 0] + ROOT["r"])
        ax_d.plot(z, m_n[:, 2], "-", color=colour, label=f"flapwise, native {label}")
        ax_d.plot(z[::2], m_c[::2, 2], marker, color=colour, mfc="none", label=f"flapwise, CalculiX S8R ({label} mesh)")
        ax_d.plot(z, m_n[:, 1], "--", color=colour)
        ax_d.plot(z[::2], m_c[::2, 1], marker, color=colour, mfc="none", ms=4)
        ax_t.plot(z, np.degrees(m_n[:, 3]), "-", color=colour, label=f"native {label}")
        ax_t.plot(z[::2], np.degrees(m_c[::2, 3]), marker, color=colour, mfc="none", label=f"CalculiX ({label} mesh)")
    ax_d.annotate("edgewise (dashed)", xy=(0.98, 0.08), xycoords="axes fraction", ha="right", fontsize=8,
                  color=TEXT_SECONDARY)
    ax_d.set_xlabel("blade radius r [m]")
    ax_d.set_ylabel("section deflection [m]")
    ax_d.set_title("Deflection along the segment", loc="left")
    ax_d.legend(fontsize=7.5)
    ax_t.axhline(0.0, color=TEXT_SECONDARY, lw=0.8)
    ax_t.set_xlabel("blade radius r [m]")
    ax_t.set_ylabel("section twist [deg]")
    ax_t.set_title("Twist along the segment", loc="left")
    ax_t.legend(fontsize=7.5)

    for side, colour, name in (("ss", SERIES[0], "suction cap"), ("ps", SERIES[1], "pressure cap")):
        z, sn, sc = cap_series(r_main, side)
        ax_s.plot(z + ROOT["r"], sn / 1e6, "-", color=colour, label=f"{name}, native Quad4")
        ax_s.plot(z[::2] + ROOT["r"], sc[::2] / 1e6, "o", color=colour, mfc="none", label=f"{name}, CalculiX S8R")
    ax_s.axhline(0.0, color=TEXT_SECONDARY, lw=0.8)
    ax_s.set_xlabel("blade radius r [m]")
    ax_s.set_ylabel("σ11 in the carbon [MPa]")
    ax_s.set_title("Spar-cap fibre stress", loc="left")
    ax_s.legend(fontsize=7.5)

    eid = r_main["panel"]
    lam = r_main["blade"].mesh.element(eid).laminate
    ref, cand = r_main["native"].profile(eid, "material"), r_main["ccx"].profile(eid, "material")
    h = lam.thickness
    zb = lam.z_interfaces[0]
    for colour, comp, name in ((SERIES[0], "s11", "σ11"), (SERIES[1], "s22", "σ22"), (SERIES[2], "t12", "τ12")):
        plies = np.array([lam.ply_index_at(z) for z in ref.z])
        for k in np.unique(plies):
            mask = plies == k
            ax_p.plot(ref.component(comp)[mask] / 1e6, (ref.z[mask] - zb) / h, color=colour,
                      label=f"{name} native" if k == plies.min() else None)
        ax_p.plot(cand.component(comp) / 1e6, (cand.z - zb) / h, "o", color=colour, mfc="none", ms=5,
                  label=f"{name} CalculiX")
    for zi in (lam.z_interfaces[1:-1] - zb) / h:
        ax_p.axhline(zi, color=TEXT_SECONDARY, lw=0.6, ls=(0, (4, 3)))
    ax_p.axvline(0.0, color=TEXT_SECONDARY, lw=0.8)
    names = " / ".join(p.material.name for p in lam.plies)
    ax_p.set_xlabel("ply stress in material axes [MPa]")
    ax_p.set_ylabel("height through the panel / thickness (inside = 0)")
    ax_p.set_title("Through the suction-side sandwich panel", loc="left")
    ax_p.legend(fontsize=7.5, loc="center right")
    fig.text(0.0, -0.02, f"IEA 15 MW blade, r = {ROOT['r']:.1f}–{TIP['r']:.1f} m, layup of the r = {ROOT['r']:.1f} m station. "
             f"Panel (bottom right, inside to outside): {names}. Dashed lines are ply interfaces.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "blade_calculix_crosscheck.png")


def main():
    if shutil.which("ccx") is None:
        print("CalculiX (ccx) is not on the PATH; skipping this example.")
        return
    results = {}
    for label, options in MODELS.items():
        t0 = time.perf_counter()
        results[label] = report(label, run(label, options))
        print(f"  ({time.perf_counter() - t0:.0f} s)")
    figure(results)


if __name__ == "__main__":
    main()
