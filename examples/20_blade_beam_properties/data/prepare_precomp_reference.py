"""PreComp properties of the example 20 section, for every cap angle in the sweep.

This script is run once; its output, ``precomp_reference.json`` next to it, is committed
so the example needs no WISDEM. To regenerate it (it takes a few seconds)::

    pip install wisdem
    python examples/20_blade_beam_properties/data/prepare_precomp_reference.py

PreComp (NREL; here WISDEM's Python port) is a thin-wall section tool: classical
lamination theory for each wall, a shear-flow model for torsion. It is given the
same section as the shell model: the NACA 0024 profile, the same sectors on each
surface (LE panel, spar cap, TE panel), the same laminae from the outer surface
inward, and the two webs, with the 2D orthotropic constants (E1, E2, G12, nu12,
density) of the two library materials.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

from blade_beam_properties import ANGLES, CHORD, REGIONS, WEBS  # noqa: E402
from composite_laminate_model import create_default_library  # noqa: E402

OUT = HERE / "precomp_reference.json"
PLY_T = 0.5e-3
SKIN = [0, 45, -45, -45, 45, 0]
WEB = [45, -45, -45, 45]
N_CAP = 16
PITCH_AXIS = 0.3


def profile(n=121):
    """NACA 0024 (closed trailing edge) in PreComp order: leading edge at (0, 0), round
    the upper surface to the trailing edge, then back along the lower surface."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    # the same closed-trailing-edge thickness law as Airfoil.naca4
    yu = 5 * 0.24 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    yu[-1] = 0.0  # one shared trailing-edge point
    yl = -yu
    return np.concatenate([x, x[-2:0:-1]]), np.concatenate([yu, yl[-2:0:-1]])


def surface(cap_angle, materials):
    """Sector boundaries and laminae of one surface; PreComp lists laminae from the
    outer surface inward, one lamina per ply here."""
    glass, carbon = materials.index("E-glass/epoxy") + 1, materials.index("T300/5208") + 1
    loc = [REGIONS[0].start, REGIONS[1].start, REGIONS[2].start, REGIONS[2].end]
    sectors = [[(glass, a) for a in SKIN], [(carbon, cap_angle)] * N_CAP, [(glass, a) for a in SKIN]]
    n_lamina = [len(s) for s in sectors]
    plies = [1] * sum(n_lamina)
    t = [PLY_T] * sum(n_lamina)
    theta = [a for s in sectors for _, a in s]
    mat = [m for s in sectors for m, _ in s]
    return loc, n_lamina, plies, t, theta, mat


def webs(materials):
    glass = materials.index("E-glass/epoxy") + 1
    loc = [w.position for w in WEBS]
    n_lamina = [len(WEB)] * len(WEBS)
    plies = [1] * (len(WEB) * len(WEBS))
    return loc, n_lamina, plies, [PLY_T] * len(plies), WEB * len(WEBS), [glass] * len(plies)


def run(cap_angle, sign):
    from wisdem.precomp import properties

    lib = create_default_library()
    materials = ["E-glass/epoxy", "T300/5208"]
    el = [lib.get(m).elastic for m in materials]
    x, y = profile()
    upper = surface(sign * cap_angle, materials)
    lower = surface(-sign * cap_angle, materials)
    loc_w, *web = webs(materials)
    out = properties.properties(
        CHORD, 0.0, 0.0, PITCH_AXIS, x, y,
        [e.E1 for e in el], [e.E2 for e in el], [e.G12 for e in el], [e.v12 for e in el],
        [lib.get(m).density for m in materials],
        *upper, *lower, len(WEBS), loc_w, *web)
    (eif, eil, gj, ea, eifl, sf, sl, sft, slt, sat, x_sc, y_sc, x_tc, y_tc, mass, area, iflap, ilag, tw,
     x_cg, y_cg) = out
    return {"angle": cap_angle, "EA": ea, "EI_flap": eif, "EI_edge": eil, "EI_flap_edge": eifl, "GJ": gj,
            "EA_flap": sf, "EA_edge": sl, "flap_GJ": sft, "edge_GJ": slt, "EA_GJ": sat,
            "x_sc": x_sc, "y_sc": y_sc, "x_tc": x_tc, "y_tc": y_tc, "x_cg": x_cg, "y_cg": y_cg, "mass": mass}


def main():
    # The cap angles go to PreComp as the shell model has them (+theta on the suction
    # side, -theta on the pressure side). PreComp documents its ply angle as measured
    # from the span axis; if its sense of rotation were the opposite of the shell
    # model's, only the sign of the coupling terms would change, not their size.
    data = {
        "notes": "PreComp (WISDEM port) on the example 20 section, chord frame. PreComp axes: x flapwise "
                 "(suction side positive), y chordwise (aft positive), from the pitch axis; bending about the "
                 "tension centre. The cap angle is the shell model's (+theta suction, -theta pressure side).",
        "sweep": [run(a, 1.0) for a in ANGLES],
    }
    OUT.write_text(json.dumps(data, indent=1, default=float))
    for r in data["sweep"]:
        print(f"{r['angle']:3d} deg: EA {r['EA']:.4g}  EI flap {r['EI_flap']:.4g}  GJ {r['GJ']:.4g}  "
              f"flap-twist {r['flap_GJ'] / np.sqrt(r['EI_flap'] * r['GJ']):+.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
