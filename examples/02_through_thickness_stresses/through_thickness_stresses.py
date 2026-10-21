"""Through-thickness stress profiles of a thick laminate: where the kinematic models differ.

A thick (a/h = 5) simply supported [0/90/0] square plate under uniform pressure.
Left: in-plane normal stress sx at the plate centre - the shear-deformable models
agree with each other, while CLPT (too stiff in shear) overshoots the peak. Right: transverse shear stress t_xz near the x = 0 support,
where shear is largest:

* FSDT gives a constant shear strain, so the stress jumps at every ply interface;
* RZT and layerwise enforce continuous interlaminar shear (they coincide here);
* CLPT recovers shear from equilibrium with a parabolic profile.

When CalculiX is installed, its 3D-expanded composite shell (C3D20R layers) is
overlaid as an independent reference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil

import numpy as np

from _shared.plotting import MODEL_COLOURS, REFERENCE, TEXT_SECONDARY, plt, save, simply_supported
from laminate_modelling_package import (
    AnalysisModel,
    CalculiXKernelAdapter,
    CLPTModel,
    ExactStressRecoverer,
    FEAMesh,
    FSDTModel,
    Laminate,
    LayerwiseModel,
    LoadCase,
    RefinedZigZagModel,
    StructuralGlobalSolver,
    create_default_library,
)
from laminate_modelling_package.laminate_mechanics import shear_transformation

A, RATIO, Q, N_ELEM = 0.1, 5, 1.0e5, 16


def element_at(mesh, x, y):
    centroids = np.array([e.coordinates.mean(axis=0)[:2] for e in mesh.elements])
    return mesh.elements[int(np.argmin(np.linalg.norm(centroids - [x, y], axis=1)))]


def main():
    card = create_default_library().get("T300/5208")
    h = A / RATIO
    lam = Laminate.from_angles(card, [0, 90, 0], h / 3)
    # Each ply sampled bottom -> top; interfaces appear twice (one value per ply).
    z_values = []
    for z_b, z_t in lam.ply_bounds():
        z_values += list(np.linspace(z_b, z_t, 9))
    z_values = np.array(z_values)
    sides = np.array([("below" if (i % 9) == 8 else "above") for i in range(len(z_values))])

    models = [CLPTModel(), FSDTModel(), RefinedZigZagModel(), LayerwiseModel()]
    centre_xy, shear_xy = (A / 2 + A / N_ELEM / 2, A / 2 + A / N_ELEM / 2), (A / N_ELEM / 2, A / 2 + A / N_ELEM / 2)
    curves = {}
    rec = ExactStressRecoverer()
    mesh0 = None
    bcs = None
    for model in models:
        mesh = FEAMesh.rectangular_plate(A, A, N_ELEM, N_ELEM, lam, model)
        bcs = simply_supported(mesh, A, A)
        result = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_pressure(Q))
        ec, es = element_at(mesh, *centre_xy), element_at(mesh, *shear_xy)
        uc, us = result.element_displacements(ec.element_id), result.element_displacements(es.element_id)
        sx = [rec.recover(ec, uc, z, side=s).stress_global[0] for z, s in zip(z_values, sides)]
        txz = [rec.recover(es, us, z, side=s).shear_global[1] for z, s in zip(z_values, sides)]
        curves[model.name] = (np.array(sx), np.array(txz))
        mesh0 = mesh if model.name == "FSDT" else mesh0

    ccx = None
    if shutil.which("ccx"):
        ec, es = element_at(mesh0, *centre_xy), element_at(mesh0, *shear_xy)
        run = CalculiXKernelAdapter().run(
            AnalysisModel(mesh0, bcs, LoadCase().add_pressure(Q), "thick_plate", (ec.element_id, es.element_id))
        )
        if run.ok:
            pc, ps = run.profile(ec.element_id, "material"), run.profile(es.element_id, "material")
            # Rotate material-axis stresses back to global axes ply by ply.
            def to_global(p):
                sx, txz = [], []
                for z, row in zip(p.z, p.values):
                    ply = lam.plies[lam.ply_index_at(z)]
                    s11, s22, t12, t23, t13 = row
                    sx.append((np.linalg.solve(ply.T, [s11, s22, t12]))[0])
                    txz.append((shear_transformation(ply.angle).T @ [t23, t13])[1])
                return p.z, np.array(sx), np.array(txz)
            ccx = (to_global(pc), to_global(ps))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
    zn = z_values / h
    for name, (sx, txz) in curves.items():
        style = dict(color=MODEL_COLOURS[name], lw=2.0, label=name)
        if name == "Layerwise":
            style.update(ls=(0, (2, 2)), lw=2.4)
        ax1.plot(sx / 1e6, zn, **style)
        ax2.plot(txz / 1e6, zn, **style)
    if ccx is not None:
        (zc, sxc, _), (zs, _, txzc) = ccx
        ax1.plot(sxc / 1e6, zc / h, "o", mfc="none", color=REFERENCE, ms=6, label="CalculiX (3D layers)")
        ax2.plot(txzc / 1e6, zs / h, "o", mfc="none", color=REFERENCE, ms=6, label="CalculiX (3D layers)")
    for ax in (ax1, ax2):
        for z in lam.z_interfaces[1:-1] / h:
            ax.axhline(z, color=TEXT_SECONDARY, lw=0.8, ls=(0, (4, 3)))
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
    ax1.set_ylabel("z / h")
    ax1.set_xlabel("in-plane stress σx at plate centre [MPa]")
    ax2.set_xlabel("transverse shear stress τxz near the x = 0 edge [MPa]")
    peak = {k: np.max(np.abs(v[0])) for k, v in curves.items()}
    ax1.set_title(f"σx: CLPT overshoots the peak by {100 * (peak['CLPT'] / peak['FSDT'] - 1):.0f}%", loc="left")
    ax2.set_title("τxz: FSDT jumps at ply interfaces", loc="left")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.06))
    for label, z in zip(("0° ply", "90° ply", "0° ply"), lam.ply_midplanes() / h):
        ax1.text(0.02, z, label, transform=ax1.get_yaxis_transform(), va="center", fontsize=8, color=TEXT_SECONDARY)
    fig.text(0.0, -0.03, f"[0/90/0] T300/5208, a/h = {RATIO}, simply supported, q = {Q/1e3:.0f} kPa, {N_ELEM} x {N_ELEM} mesh. "
             "Dashed horizontal lines mark ply interfaces.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "through_thickness_stresses.png")


if __name__ == "__main__":
    main()
