"""Sandwich panel: where zig-zag and layerwise theories earn their keep.

A simply supported 400 mm square sandwich panel has thin carbon skins on a
PVC foam core. The core is about 1700 times softer than the skins, so most of
the transverse shear deformation happens in the core. FSDT smears the shear
stiffness over the whole thickness and is far too stiff; RZT and layerwise
follow the ply-by-ply shear compliance and agree with classical sandwich theory.

Reference: the Navier solution with the sandwich shear stiffness
S = G_core d^2 / c (d = distance between skin centroids, c = core thickness).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.analytical import navier_clpt_centre, navier_fsdt_centre
from _shared.plotting import MODEL_COLOURS, REFERENCE, TEXT_SECONDARY, plt, save, simply_supported
from laminate_modelling_package import (
    CLPTModel,
    ElasticProperties3D,
    ExactStressRecoverer,
    FEAMesh,
    FSDTModel,
    HygrothermalProperties,
    Laminate,
    LayerwiseModel,
    LoadCase,
    MaterialCard,
    Ply,
    RefinedZigZagModel,
    StrengthProperties3D,
    StructuralGlobalSolver,
    create_default_library,
    integrate_abd,
)

A, Q, N_ELEM, PLY_T = 0.4, 1.0e4, 12, 0.25e-3
CORES = np.array([5, 10, 15, 20, 30, 40, 60]) * 1e-3
PROFILE_CORE = 20e-3

FOAM = MaterialCard(
    name="PVC foam H100",
    elastic=ElasticProperties3D.isotropic(105e6, 0.3),
    strength=StrengthProperties3D(3.1e6, 1.9e6, 3.1e6, 1.9e6, 1.6e6),
    hygrothermal=HygrothermalProperties(alpha1=35e-6, alpha2=35e-6),
    density=100.0,
    description="Cross-linked PVC foam core (typical datasheet values)",
)


def sandwich(skin, core_thickness):
    plies = [Ply(skin, PLY_T, 0), Ply(skin, PLY_T, 90), Ply(FOAM, core_thickness, 0),
             Ply(skin, PLY_T, 90), Ply(skin, PLY_T, 0)]
    return Laminate(plies, name=f"sandwich c = {core_thickness * 1e3:.0f} mm")


def sandwich_theory(lam, core_thickness):
    D = integrate_abd(lam)[3:, 3:]
    d = core_thickness + 2 * PLY_T
    S = FOAM.elastic.G13 * d**2 / core_thickness
    return navier_fsdt_centre(D, np.diag([S, S]), A, A, Q)


def solve(lam, model):
    mesh = FEAMesh.rectangular_plate(A, A, N_ELEM, N_ELEM, lam, model)
    return mesh, StructuralGlobalSolver().solve(mesh, simply_supported(mesh, A, A), LoadCase().add_pressure(Q))


def main():
    skin = create_default_library().get("T300/5208")
    models = [CLPTModel(), FSDTModel(), RefinedZigZagModel(), LayerwiseModel()]
    w = {m.name: [] for m in models}
    reference = []
    print(f"{'core [mm]':>9}" + "".join(f"{m.name:>11}" for m in models) + f"{'sandwich':>11}   w_centre [mm]")
    for c in CORES:
        lam = sandwich(skin, c)
        for model in models:
            mesh, result = solve(lam, model)
            w[model.name].append(result.node_displacement(mesh.closest_node((A / 2, A / 2)))[2])
        reference.append(sandwich_theory(lam, c))
        print(f"{c * 1e3:9.0f}" + "".join(f"{w[m.name][-1] * 1e3:11.3f}" for m in models) + f"{reference[-1] * 1e3:11.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1.0]})
    for name, values in w.items():
        style = dict(marker="o", ms=6, ls="none", color=MODEL_COLOURS[name], label=name)
        if name == "Layerwise":
            style.update(mfc="none", ms=10, mew=1.6)
        ax1.plot(CORES * 1e3, np.array(values) * 1e3, **style)
    ax1.plot(CORES * 1e3, np.array(reference) * 1e3, color=REFERENCE, lw=1.2, ls=(0, (4, 3)),
             label="sandwich theory (Navier)")
    ax1.set_yscale("log")
    ax1.set_xlabel("core thickness c [mm]")
    ax1.set_ylabel("centre deflection [mm]")
    i = list(CORES).index(PROFILE_CORE)
    ratio = w["RZT"][i] / w["FSDT"][i]
    ax1.set_title(f"FSDT underestimates deflection {ratio:.1f}x at c = {PROFILE_CORE * 1e3:.0f} mm")
    ax1.legend(loc="upper right")

    lam = sandwich(skin, PROFILE_CORE)
    h = lam.thickness
    rec = ExactStressRecoverer()
    for model in models[1:]:
        mesh, result = solve(lam, model)
        centroids = np.array([e.coordinates.mean(axis=0)[:2] for e in mesh.elements])
        element = mesh.elements[int(np.argmin(np.linalg.norm(centroids - [A / N_ELEM / 2, A / 2], axis=1)))]
        u = result.element_displacements(element.element_id)
        z, tau = [], []
        for zb, zt in lam.ply_bounds():
            for zz in np.linspace(zb, zt, 9):
                side = "below" if np.isclose(zz, zt) else "above"
                z.append(zz)
                tau.append(rec.recover(element, u, zz, side=side).shear_global[1])
        style = dict(color=MODEL_COLOURS[model.name], label=model.name)
        if model.name == "Layerwise":
            style.update(ls=(0, (2, 2)), lw=2.4)
        ax2.plot(np.array(tau) / 1e3, np.array(z) / h, **style)
    for zi in lam.z_interfaces[1:-1] / h:
        ax2.axhline(zi, color=TEXT_SECONDARY, lw=0.6, ls=(0, (4, 3)))
    ax2.set_xscale("log")
    ax2.set_xlabel("τxz near the x = 0 edge [kPa, log scale]")
    ax2.set_ylabel("z / h")
    ax2.set_title("Shear in the core: FSDT puts it in the skins")
    ax2.legend(loc="center left", bbox_to_anchor=(0.45, 0.7))
    fig.text(0.0, -0.03, f"Skins [0/90] and [90/0] T300/5208 ({2 * PLY_T * 1e3:.1f} mm each), PVC foam core (E = 105 MPa), "
             f"{A * 1e3:.0f} mm square, simply supported, q = {Q / 1e3:.0f} kPa, {N_ELEM} x {N_ELEM} mesh.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "sandwich_panel.png")


if __name__ == "__main__":
    main()
