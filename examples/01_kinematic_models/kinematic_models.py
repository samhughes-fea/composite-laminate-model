"""How much does shear deformation matter? Centre deflection versus span-to-thickness ratio.

A simply supported [0/90/90/0] T300/5208 square plate under uniform pressure is
solved with each kinematic model on the same 12 x 12 mesh, for a/h from 4 (thick)
to 100 (thin). Deflections are normalised as w_bar = 100 w E2 h^3 / (q a^4) and
compared with the Navier series solutions of CLPT and FSDT.

Reading: CLPT is independent of a/h and underestimates the deflection of thick
plates; FSDT, RZT and layerwise converge to CLPT as the plate gets thin.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.analytical import navier_clpt_centre, navier_fsdt_centre
from _shared.plotting import MODEL_COLOURS, REFERENCE, TEXT_SECONDARY, plt, save, simply_supported
from composite_laminate_model import (
    CLPTModel,
    FEAMesh,
    FSDTModel,
    Laminate,
    LayerwiseModel,
    LoadCase,
    RefinedZigZagModel,
    StructuralGlobalSolver,
    create_default_library,
    integrate_abd,
)

A, Q, N_ELEM = 0.2, 1.0e4, 12
RATIOS = np.array([4, 5, 6, 8, 10, 15, 20, 30, 50, 100])


def main():
    card = create_default_library().get("T300/5208")
    E2 = card.elastic.E2
    models = [CLPTModel(), FSDTModel(), RefinedZigZagModel(), LayerwiseModel()]
    fe = {m.name: [] for m in models}
    ref_clpt, ref_fsdt = [], []
    solver = StructuralGlobalSolver()
    for ratio in RATIOS:
        h = A / ratio
        lam = Laminate.from_angles(card, [0, 90, 90, 0], h / 4)
        scale = 100 * E2 * h**3 / (Q * A**4)
        D = integrate_abd(lam)[3:, 3:]
        ref_clpt.append(navier_clpt_centre(D, A, A, Q) * scale)
        ref_fsdt.append(navier_fsdt_centre(D, FSDTModel().transverse_shear_matrix(lam), A, A, Q) * scale)
        for model in models:
            mesh = FEAMesh.rectangular_plate(A, A, N_ELEM, N_ELEM, lam, model)
            result = solver.solve(mesh, simply_supported(mesh, A, A), LoadCase().add_pressure(Q))
            fe[model.name].append(result.node_displacement(mesh.closest_node((A / 2, A / 2)))[2] * scale)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    markers = {"CLPT": "o", "FSDT": "s", "RZT": "^", "Layerwise": "v"}
    for name, values in fe.items():
        hollow = name == "Layerwise"
        ax.plot(RATIOS, values, ls="none", marker=markers[name], ms=9 if hollow else 7,
                mfc="none" if hollow else MODEL_COLOURS[name], mew=1.6, color=MODEL_COLOURS[name], label=f"FE {name}", zorder=3)
    ax.plot(RATIOS, ref_clpt, ls="--", lw=1.2, color=REFERENCE, label="Navier CLPT", zorder=4)
    ax.plot(RATIOS, ref_fsdt, ls=":", lw=1.5, color=REFERENCE, label="Navier FSDT (k = 5/6)", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks(RATIOS)
    ax.set_xticklabels([str(r) for r in RATIOS])
    ax.minorticks_off()
    ax.set_xlabel("span-to-thickness ratio a/h")
    ax.set_ylabel("normalised centre deflection  100 w E2 h³ / (q a⁴)")
    ax.set_title("Shear deformation multiplies the deflection by {:.1f} at a/h = 4".format(fe["FSDT"][0] / fe["CLPT"][0]))
    ax.text(RATIOS[-1], fe["CLPT"][-1] * 1.04, "all models converge\nfor thin plates", ha="right", va="bottom", color=TEXT_SECONDARY, fontsize=8.5)
    ax.set_ylim(0, max(max(v) for v in fe.values()) * 1.12)
    ax.legend(loc="upper right", ncol=2)
    ax.text(0.0, -0.2, "[0/90/90/0] T300/5208, simply supported, uniform pressure, 12 x 12 Quad4 mesh. "
            "RZT and layerwise share the same condensed shear stiffness, so their points coincide.",
            transform=ax.transAxes, fontsize=7.5, color=TEXT_SECONDARY, wrap=True)
    save(fig, __file__, "kinematic_models_vs_thickness.png")
    for i, r in enumerate(RATIOS):
        print(f"a/h={r:>4}: " + "  ".join(f"{k}={v[i]:.4f}" for k, v in fe.items()) + f"  Navier CLPT={ref_clpt[i]:.4f} FSDT={ref_fsdt[i]:.4f}")


if __name__ == "__main__":
    main()
