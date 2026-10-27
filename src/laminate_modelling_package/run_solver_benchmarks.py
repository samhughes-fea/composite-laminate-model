#!/usr/bin/env python3
"""Run the same asymmetric laminate plate through every FEA kernel and compare.

The model is a simply supported square plate with an asymmetric carbon/epoxy
stack, loaded by a uniform pressure plus a uniform cool-down. Each kernel
(native NumPy, CalculiX, Nastran) is timed with ``time.perf_counter``; kernels
whose executable is missing write their input deck and are reported as skipped.

Usage:
    laminate-benchmarks [--n 10] [--pressure 5000] [--delta-t -50]
                                    [--work-dir benchmark_runs] [--ccx ccx] [--nastran nastran]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from laminate_modelling_package import (
    DOF,
    AnalysisModel,
    BoundaryConditions,
    CalculiXKernelAdapter,
    CLPTModel,
    FEAMesh,
    FSDTModel,
    HashinCriterion,
    HygrothermalState,
    KernelResult,
    KernelStatus,
    Laminate,
    LayerwiseModel,
    LoadCase,
    MaximumStressCriterion,
    NastranKernelAdapter,
    NativePythonKernelAdapter,
    RefinedZigZagModel,
    StressDistributionVerifier,
    StructuralGlobalSolver,
    TsaiWuCriterion,
    create_default_library,
)

ASYMMETRIC_STACK = (0.0, 45.0, -45.0, 90.0, 0.0, 90.0)
PLY_THICKNESS = 0.25e-3  # m
PLATE_SIZE = 0.2  # m


def build_model(n: int, pressure: float, delta_t: float, material: str = "T300/5208") -> AnalysisModel:
    card = create_default_library().get(material)
    laminate = Laminate.from_angles(card, ASYMMETRIC_STACK, PLY_THICKNESS)
    mesh = FEAMesh.rectangular_plate(PLATE_SIZE, PLATE_SIZE, n, n, laminate, FSDTModel())

    bcs = BoundaryConditions()
    a = PLATE_SIZE
    edges = set(mesh.nodes_on_line(0, 0.0) + mesh.nodes_on_line(0, a) + mesh.nodes_on_line(1, 0.0) + mesh.nodes_on_line(1, a))
    bcs.fix(sorted(edges), [DOF.UZ])  # simply supported: w = 0 on all edges
    bcs.fix([mesh.closest_node((0.0, 0.0))], [DOF.UX, DOF.UY])  # remove in-plane rigid-body motion
    bcs.fix([mesh.closest_node((a, 0.0))], [DOF.UY])

    loads = LoadCase("pressure+cooldown").add_pressure(pressure)
    loads.set_hygrothermal(HygrothermalState(delta_temperature=delta_t))

    # Element at (or, for even n, touching) the plate centre: row n//2, column n//2.
    centre_element = mesh.element_ids[(n // 2) * n + n // 2]
    return AnalysisModel(mesh, bcs, loads, title="asymmetric_plate", monitor_elements=(centre_element,))


def format_table(header: Sequence[str], rows: List[Sequence[str]]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip()
    return "\n".join([line(header), "  ".join("-" * w for w in widths), *map(line, rows)])


def summary_matrix(results: Dict[str, KernelResult]) -> str:
    header = ["kernel", "status", "max|UX| [m]", "max|UY| [m]", "max|UZ| [m]", "runtime [s]", "note"]
    rows = []
    for name, r in results.items():
        if r.ok:
            values = [f"{r.max_abs_displacement(d):.5e}" for d in (0, 1, 2)]
        else:
            values = ["-", "-", "-"]
        note = r.message.splitlines()[0] if r.message else ""
        rows.append([name, r.status.value, *values, f"{r.runtime:.4f}", note[:70]])
    return format_table(header, rows)


def kinematic_study(model: AnalysisModel) -> str:
    """Centre deflection of the native solver for each kinematic model."""
    solver = StructuralGlobalSolver()
    centre = model.mesh.closest_node((PLATE_SIZE / 2, PLATE_SIZE / 2))
    rows = []
    for kinematics in (CLPTModel(), FSDTModel(), RefinedZigZagModel(), LayerwiseModel()):
        mesh = model.mesh.copy()
        mesh.set_kinematics(kinematics)
        start = time.perf_counter()
        solution = solver.solve(mesh, model.boundary_conditions, model.load_case)
        rows.append([kinematics.name, f"{solution.node_displacement(centre)[2]:.5e}", f"{time.perf_counter() - start:.4f}"])
    return format_table(["kinematics", "w(centre) [m]", "runtime [s]"], rows)


def failure_summary(model: AnalysisModel, native: KernelResult) -> str:
    element_id = model.monitor_elements[0]
    profile = native.profile(element_id, "material")
    laminate = model.mesh.element(element_id).laminate
    rows = []
    for criterion in (MaximumStressCriterion(), TsaiWuCriterion(), HashinCriterion("3d")):
        worst, where = None, None
        for z, (s11, s22, t12, t23, t13) in zip(profile.z, profile.values):
            ply = laminate.ply_index_at(z)
            stress = np.array([s11, s22, 0.0, t23, t13, t12])
            result = criterion.evaluate(stress, laminate.plies[ply].material.strength)
            if worst is None or result.index > worst.index:
                worst, where = result, ply
        rows.append([criterion.name, f"{worst.index:.4f}", f"{worst.strength_ratio:.3f}", worst.mode.value, str(where + 1)])
    return format_table(["criterion", "max index", "strength ratio", "mode", "ply"], rows)


def verification_report(model: AnalysisModel, results: Dict[str, KernelResult]) -> str:
    verifier = StressDistributionVerifier()
    native = results["native"]
    blocks = []
    for name, result in results.items():
        if name == "native" or not result.ok:
            continue
        stats = list(verifier.compare_displacements(native.displacements, result.displacements).values())
        blocks.append(verifier.format_table(stats, f"Displacements: {name} vs native"))
        for eid in model.monitor_elements:
            for frame in ("material", "material_midply", "global"):
                ref, cand = native.profile(eid, frame), result.profile(eid, frame)
                if ref is None or cand is None:
                    continue
                profile_stats = verifier.compare_profiles(ref, cand, components=("s11", "s22", "t12"))
                blocks.append(
                    verifier.format_table(profile_stats.values(), f"Ply stresses ({frame}, element {eid}): {name} vs native")
                )
    return "\n\n".join(blocks) if blocks else "No external kernel completed; nothing to cross-verify."


def main(argv: Sequence[str] = None) -> Dict[str, KernelResult]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="elements per side")
    parser.add_argument("--pressure", type=float, default=1.0e3, help="uniform pressure [Pa]")
    parser.add_argument("--delta-t", type=float, default=-50.0, help="uniform temperature change [K]")
    parser.add_argument("--work-dir", type=Path, default=Path("benchmark_runs"), help="directory for solver decks")
    parser.add_argument("--ccx", default="ccx", help="CalculiX executable")
    parser.add_argument("--nastran", default="nastran", help="Nastran executable")
    args = parser.parse_args(argv)

    model = build_model(args.n, args.pressure, args.delta_t)
    laminate = model.mesh.elements[0].laminate
    print("=" * 78)
    print("Composite laminate solver benchmark")
    print(f"  laminate : {laminate.stacking_sequence} T300/5208, h = {laminate.thickness * 1e3:.2f} mm (asymmetric)")
    print(f"  mesh     : {args.n} x {args.n} Quad4 on {PLATE_SIZE} m x {PLATE_SIZE} m, {model.mesh.n_dof} DOF")
    print(f"  loading  : p = {args.pressure:g} Pa, dT = {args.delta_t:g} K, simply supported edges")
    print("=" * 78)

    kernels = [
        NativePythonKernelAdapter(),
        CalculiXKernelAdapter(executable=args.ccx, work_dir=args.work_dir / "calculix"),
        NastranKernelAdapter(executable=args.nastran, work_dir=args.work_dir / "nastran"),
    ]
    results: Dict[str, KernelResult] = {}
    for kernel in kernels:
        print(f"running {kernel.name:<9} ...", end=" ", flush=True)
        results[kernel.name] = kernel.run(model)
        print(f"{results[kernel.name].status.value} ({results[kernel.name].runtime:.3f} s)")

    print("\nBenchmark summary")
    print(summary_matrix(results))
    print("\nNative kinematic models (same mesh and loads)")
    print(kinematic_study(model))
    if results["native"].status is KernelStatus.SUCCESS:
        print("\nFailure indices at the centre element (native)")
        print(failure_summary(model, results["native"]))
    print("\nCross-verification")
    print(verification_report(model, results))
    for name, result in results.items():
        if "deck" in result.files:
            print(f"\n{name} deck: {result.files['deck']}")
    return results


def cli() -> int:
    """Console-script entry point (``laminate-benchmarks``)."""
    results = main()
    return 0 if results["native"].ok else 1


if __name__ == "__main__":
    raise SystemExit(cli())
