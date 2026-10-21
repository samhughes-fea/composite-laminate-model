"""Solver scaling: runtime against model size for each solver path.

The same simply supported [0/45/-45/90]s plate under pressure is solved on
meshes from 4 x 4 to 48 x 48 elements with:

* the native solver with the sparse SuperLU back end (SciPy, the default),
* the native solver with the dense Cholesky back end (NumPy only; stopped at
  24 x 24 because its memory grows with the square of the DOF count),
* CalculiX through ``CalculiXKernelAdapter`` (deck writing, the ccx run and
  result parsing included), when ``ccx`` is installed.

The native time is split into assembly (element stiffness matrices, written in
Python) and the linear solve, which shows where the time goes.
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save, simply_supported
from laminate_modelling_package import (
    AnalysisModel,
    CalculiXKernelAdapter,
    DenseLinearSolver,
    FEAMesh,
    FSDTModel,
    GlobalAssembler,
    Laminate,
    LoadCase,
    SparseLinearSolver,
    StructuralGlobalSolver,
    create_default_library,
)

A = 0.2
MESHES = [4, 8, 12, 16, 24, 32, 40, 48]
DENSE_LIMIT = 24


def model(lam, n):
    mesh = FEAMesh.rectangular_plate(A, A, n, n, lam, FSDTModel())
    return mesh, simply_supported(mesh, A, A), LoadCase().add_pressure(1.0e3)


def timed(fn, repeats=1):
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - start)
    return best, out


def main():
    card = create_default_library().get("T300/5208")
    lam = Laminate.from_angles(card, [0, 45, -45, 90, 90, -45, 45, 0], 0.125e-3)
    has_ccx = shutil.which("ccx") is not None
    rows = []
    print(f"{'mesh':>6}{'DOF':>8}{'assembly':>10}{'sparse':>10}{'dense':>10}{'ccx':>10}   [s]")
    with tempfile.TemporaryDirectory() as work:
        for n in MESHES:
            mesh, bcs, loads = model(lam, n)
            t_asm, _ = timed(lambda: GlobalAssembler().assemble_stiffness(mesh))
            t_sparse, sol = timed(lambda: StructuralGlobalSolver(linear_solver=SparseLinearSolver()).solve(mesh, bcs, loads))
            t_dense = timed(lambda: StructuralGlobalSolver(linear_solver=DenseLinearSolver()).solve(mesh, bcs, loads))[0] \
                if n <= DENSE_LIMIT else np.nan
            t_ccx = np.nan
            if has_ccx:
                run = CalculiXKernelAdapter(work_dir=Path(work) / f"n{n}").run(AnalysisModel(mesh, bcs, loads, f"plate{n}"))
                t_ccx = run.runtime if run.ok else np.nan
            rows.append((n, mesh.n_dof, t_asm, t_sparse, t_dense, t_ccx))
            print(f"{n:6d}{mesh.n_dof:8d}{t_asm:10.3f}{t_sparse:10.3f}{t_dense:10.3f}{t_ccx:10.3f}")
    rows = np.array(rows)
    dof = rows[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax1.loglog(dof, rows[:, 3], marker="o", color=SERIES[0], label="native, sparse (SuperLU)")
    ax1.loglog(dof, rows[:, 4], marker="s", color=SERIES[1], label="native, dense (Cholesky)")
    if has_ccx:
        ax1.loglog(dof, rows[:, 5], marker="^", color=SERIES[2], label="CalculiX (incl. file I/O)")
    ax1.set_xlabel("model degrees of freedom")
    ax1.set_ylabel("wall time [s]")
    ax1.set_title("Sparse native solves scale almost linearly")
    ax1.legend(loc="upper left")

    solve_only = rows[:, 3] - rows[:, 2]
    ax2.bar(np.arange(len(MESHES)), rows[:, 2], color=SERIES[0], label="assembly (Python element loop)")
    ax2.bar(np.arange(len(MESHES)), solve_only, bottom=rows[:, 2], color=SERIES[3],
            label="constraints, solve and recovery", edgecolor="#fcfcfb", linewidth=1)
    ax2.set_xticks(np.arange(len(MESHES)))
    ax2.set_xticklabels([f"{n}²" for n in MESHES])
    ax2.set_xlabel("elements")
    ax2.set_ylabel("native sparse wall time [s]")
    share = rows[-1, 2] / rows[-1, 3] * 100
    ax2.set_title(f"Assembly is {share:.0f}% of the time at {MESHES[-1]}²")
    ax2.grid(axis="x", visible=False)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left")
    fig.text(0.0, -0.03, "Simply supported [0/45/-45/90]s T300/5208 plate under pressure, FSDT. Single runs on the "
             "machine that built the figure; absolute times vary by machine.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "solver_scaling.png")


if __name__ == "__main__":
    main()
