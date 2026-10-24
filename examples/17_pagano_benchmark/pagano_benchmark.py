"""Shell theories against Pagano's exact 3D elasticity solution.

Pagano (1970) solved a simply supported [0/90/0] square plate under a doubly
sinusoidal pressure exactly, with no assumption through the thickness. It is the
standard benchmark for laminated plate theories. ``PaganoPlate`` reproduces it
with a transfer-matrix method.

Left: centre deflection error against span/thickness S for CLPT, FSDT, RZT and
layerwise. Middle and right: interlaminar shear near a support and the
through-thickness normal stress at the centre for a thick plate (S = 4), from
``EquilibriumStressRecoverer`` applied to the FSDT and RZT solutions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import MODEL_COLOURS, REFERENCE, TEXT_SECONDARY, plt, save, simply_supported
from laminate_modelling_package import (
    CLPTModel,
    ElasticProperties3D,
    EquilibriumStressRecoverer,
    FSDTModel,
    Laminate,
    LayerwiseModel,
    LoadCase,
    MaterialCard,
    PaganoPlate,
    RectangularPlateBuilder,
    RefinedZigZagModel,
    StrengthProperties3D,
    StructuralGlobalSolver,
)

H, N = 0.01, 16
RATIOS = [4, 6, 10, 20, 50, 100]
E2 = 1.0e9
CARD = MaterialCard(
    "Pagano",
    ElasticProperties3D(E1=25 * E2, E2=E2, E3=E2, v12=0.25, v13=0.25, v23=0.25, G12=0.5 * E2, G13=0.5 * E2, G23=0.2 * E2),
    StrengthProperties3D(1e9, 1e9, 1e8, 1e8, 1e8),
)
MODELS = {
    "CLPT": CLPTModel(),
    "FSDT": FSDTModel(),
    "RZT": RefinedZigZagModel(),
    "Layerwise": LayerwiseModel(sublayers=3),
}


def solve(model, S):
    a = S * H
    lam = Laminate.from_angles(CARD, [0, 90, 0], H / 3)
    mesh = RectangularPlateBuilder(a, a, N, N).build(lam, model)
    loads = LoadCase().add_pressure(lambda x: np.sin(np.pi * x[0] / a) * np.sin(np.pi * x[1] / a))
    result = StructuralGlobalSolver().solve(mesh, simply_supported(mesh, a, a), loads)
    return lam, a, mesh, loads, result


def main():
    errors = {name: [] for name in MODELS}
    print(f"{'S':>4}{'Pagano w':>10}" + "".join(f"{n:>11}" for n in MODELS))
    for S in RATIOS:
        exact = None
        row = []
        for name, model in MODELS.items():
            lam, a, mesh, _, result = solve(model, S)
            if exact is None:
                exact = PaganoPlate(lam, a, a).centre_deflection()
            w = result.node_displacement(mesh.closest_node((a / 2, a / 2)))[2]
            errors[name].append(100 * (w / exact - 1))
            row.append(errors[name][-1])
        wbar = 100 * E2 * H**3 * exact / (S * H) ** 4
        print(f"{S:>4}{wbar:>10.4f}" + "".join(f"{e:>10.2f}%" for e in row))

    # Through-thickness profiles at S = 4
    S = 4
    profiles = {}
    for name in ("FSDT", "RZT"):
        lam, a, mesh, loads, result = solve(MODELS[name], S)
        rec = EquilibriumStressRecoverer(result, loads)
        edge = min(mesh.elements, key=lambda e: np.hypot(e.centroid[0], e.centroid[1] - a / 2))
        centre = min(mesh.elements, key=lambda e: np.hypot(e.centroid[0] - a / 2, e.centroid[1] - a / 2))
        z = np.linspace(lam.z_interfaces[0], lam.z_interfaces[-1], 61)
        tau = np.array([rec.transverse_stresses(edge, 0.0, 0.0, zz)[1] for zz in z])
        sz = np.array([rec.transverse_stresses(centre, 0.0, 0.0, zz)[2] for zz in z])
        profiles[name] = (z, tau, sz, edge.centroid[:2], centre.centroid[:2])
    ref = PaganoPlate(lam, a, a).profile(41)
    (xe, ye), (xc, yc) = profiles["RZT"][3], profiles["RZT"][4]
    tau_ref = ref.tau_xz * np.cos(np.pi * xe / a) * np.sin(np.pi * ye / a)
    sz_ref = ref.sigma_z * np.sin(np.pi * xc / a) * np.sin(np.pi * yc / a)
    mid = len(profiles["RZT"][0]) // 2
    print(f"S = 4, mid-plane tau_xz near the support: Pagano {np.interp(0.0, ref.z, tau_ref):.4f}, "
          f"FSDT {profiles['FSDT'][1][mid]:.4f}, RZT {profiles['RZT'][1][mid]:.4f}")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.0, 4.5), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    for name, err in errors.items():
        style = dict(color=MODEL_COLOURS[name], marker="o", ms=5, label=name)
        if name == "Layerwise":
            style.update(ls=(0, (2, 2)), mfc="none", ms=8)
        ax0.plot(RATIOS, err, **style)
    ax0.axhline(0.0, color=REFERENCE, lw=1.0, label="Pagano (exact 3D)")
    ax0.set_xscale("log")
    ax0.set_xticks(RATIOS)
    ax0.set_xticklabels([str(r) for r in RATIOS])
    ax0.set_xlabel("span / thickness, S = a / h")
    ax0.set_ylabel("centre deflection error vs 3D [%]")
    ax0.set_title("Zig-zag and layerwise stay within 2%", loc="left")
    ax0.legend(loc="lower right")

    for ax, column, ref_values, xlabel, title in (
        (ax1, 1, tau_ref, "τxz near the x = 0 support [q0 units]", "Interlaminar shear, S = 4"),
        (ax2, 2, sz_ref, "σz at the plate centre [q0 units]", "Transverse normal stress, S = 4"),
    ):
        ax.plot(ref_values, ref.z / H, color=REFERENCE, lw=1.0, ls=(0, (4, 2)), label="Pagano (exact 3D)")
        for name in ("FSDT", "RZT"):
            z, *vals = profiles[name][:3]
            ax.plot(vals[column - 1], z / H, color=MODEL_COLOURS[name], label=f"{name} + equilibrium")
        for zi in lam.z_interfaces[1:-1] / H:
            ax.axhline(zi, color=TEXT_SECONDARY, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
    ax1.set_ylabel("z / h")
    ax1.legend(loc="lower right", fontsize=8)
    fig.text(0.0, -0.04, f"[0/90/0] Pagano material (E1/E2 = 25), simply supported, q = q0 sin(πx/a) sin(πy/a), "
             f"{N} x {N} curved Quad8, layerwise with 3 sub-layers per ply. Dashed horizontal lines mark ply interfaces.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "pagano_benchmark.png")


if __name__ == "__main__":
    main()
