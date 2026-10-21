"""Cross-check the native solver against CalculiX on the benchmark plate.

The model is the one ``laminate-benchmarks`` builds: a simply supported
200 mm square plate with the asymmetric stack [0/45/-45/90/0/90] in T300/5208,
loaded by 1 kPa pressure and a 50 K cool-down. The same ``AnalysisModel`` is
run through ``NativePythonKernelAdapter`` and ``CalculiXKernelAdapter``.

* Left: every shared node's deflection, CalculiX against native. Points on the
  diagonal mean the two solvers agree.
* Right: in-plane ply stresses in fibre axes through the thickness of the
  centre element. CalculiX reports layer stresses at its own integration
  points, so they are drawn as markers over the native curves.

CalculiX must be on the PATH as ``ccx`` (for example ``apt install calculix-ccx``).
Without it the script prints a message and exits without a figure.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from laminate_modelling_package import CalculiXKernelAdapter, NativePythonKernelAdapter, StressDistributionVerifier
from run_solver_benchmarks import build_model

WORK_DIR = Path(__file__).resolve().parent / "calculix_run"


def main():
    if shutil.which("ccx") is None:
        print("CalculiX (ccx) is not on the PATH; skipping this example.")
        return
    model = build_model(n=10, pressure=1.0e3, delta_t=-50.0)
    native = NativePythonKernelAdapter().run(model)
    ccx = CalculiXKernelAdapter(work_dir=WORK_DIR).run(model)
    if not ccx.ok:
        print(f"CalculiX run failed: {ccx.message}")
        return

    verifier = StressDistributionVerifier()
    stats = verifier.compare_displacements(native.displacements, ccx.displacements)
    print(verifier.format_table(stats.values(), "Displacements: CalculiX vs native"))
    eid = model.monitor_elements[0]
    ref, cand = native.profile(eid, "material"), ccx.profile(eid, "material")
    profile_stats = verifier.compare_profiles(ref, cand, components=("s11", "s22", "t12"))
    print(verifier.format_table(profile_stats.values(), f"Ply stresses (material axes, element {eid})"))
    print(f"runtime: native {native.runtime:.3f} s, CalculiX {ccx.runtime:.3f} s")

    common = [n for n in native.displacements if n in ccx.displacements]
    w_native = np.array([native.displacements[n][2] for n in common]) * 1e3
    w_ccx = np.array([ccx.displacements[n][2] for n in common]) * 1e3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.0, 1.25]})
    lim = 1.05 * max(np.abs(w_native).max(), np.abs(w_ccx).max())
    ax1.plot([-lim, lim], [-lim, lim], ls=(0, (4, 3)), lw=1.0, color=REFERENCE, label="perfect agreement")
    ax1.plot(w_native, w_ccx, "o", ms=4, color=SERIES[0], alpha=0.8, label=f"{len(common)} nodes")
    low = min(0.0, w_native.min(), w_ccx.min()) - 0.05 * lim
    ax1.set_xlim(low, lim)
    ax1.set_ylim(low, lim)
    ax1.set_aspect("equal")
    ax1.set_xlabel("w, native [mm]")
    ax1.set_ylabel("w, CalculiX S8R [mm]")
    uz = stats["UZ"]
    ax1.set_title(f"Deflections agree within {uz.normalized_max_error * 100:.1f}% of the peak")
    ax1.legend(loc="upper left")

    laminate = model.mesh.element(eid).laminate
    h = laminate.thickness
    for colour, name, label in ((SERIES[0], "s11", "σ11"), (SERIES[1], "s22", "σ22"), (SERIES[2], "t12", "τ12")):
        plies = np.array([laminate.ply_index_at(z) for z in ref.z])
        for k in np.unique(plies):  # one segment per ply: stresses jump at interfaces
            mask = plies == k
            order = np.argsort(ref.z[mask], kind="stable")
            ax2.plot(ref.component(name)[mask][order] / 1e6, ref.z[mask][order] / h, color=colour,
                     label=f"{label} native" if k == plies.min() else None)
        ax2.plot(cand.component(name) / 1e6, cand.z / h, "o", mfc="none", ms=6, color=colour, label=f"{label} CalculiX")
    for z in laminate.z_interfaces[1:-1] / h:
        ax2.axhline(z, color=TEXT_SECONDARY, lw=0.6, ls=(0, (4, 3)))
    ax2.axvline(0, color=TEXT_SECONDARY, lw=0.8)
    peak = max(s.reference_peak for s in profile_stats.values())
    worst = max(s.max_absolute_error for s in profile_stats.values()) / peak
    ax2.set_title(f"Ply stresses agree within {worst * 100:.1f}% of the largest stress")
    ax2.set_xlabel("stress in fibre axes at the centre element [MPa]")
    ax2.set_ylabel("z / h")
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.text(0.0, -0.03, "[0/45/-45/90/0/90] T300/5208, 200 mm square, h = 1.5 mm, simply supported, p = 1 kPa, ΔT = -50 K, "
             "10 x 10 mesh. Dashed lines mark ply interfaces.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "calculix_crosscheck.png")


if __name__ == "__main__":
    main()
