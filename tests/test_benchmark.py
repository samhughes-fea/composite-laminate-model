import numpy as np
import pytest

from composite_laminate_model import run_solver_benchmarks
from composite_laminate_model import KernelStatus, StressDistributionVerifier, ThicknessProfile


def test_array_statistics():
    stats = StressDistributionVerifier().compare_arrays([1.0, -2.0, 4.0], [1.5, -2.0, 3.0], "x")
    assert stats.mean_absolute_error == pytest.approx(0.5)
    assert stats.max_absolute_error == pytest.approx(1.0)
    assert stats.rms_error == pytest.approx(np.sqrt((0.25 + 1.0) / 3))
    assert stats.normalized_max_error == pytest.approx(0.25)
    with pytest.raises(ValueError):
        StressDistributionVerifier().compare_arrays([1.0], [1.0, 2.0])


def test_profile_comparison_interpolates_candidate():
    ref = ThicknessProfile(1, [-1.0, 0.0, 1.0], [[0.0], [1.0], [2.0]], ("s11",))
    cand = ThicknessProfile(1, [-1.0, 1.0], [[0.0], [2.2]], ("s11",))
    stats = StressDistributionVerifier().compare_profiles(ref, cand)
    assert stats["s11"].max_absolute_error == pytest.approx(0.2)
    with pytest.raises(ValueError):
        ThicknessProfile(1, [0.0, 1.0], [[1.0, 2.0]], ("a", "b"))
    with pytest.raises(ValueError):
        StressDistributionVerifier().compare_profiles(ref, ThicknessProfile(1, [0.0], [[1.0]], ("s11",), frame="material"))


def test_displacement_comparison_skips_missing_dofs():
    ref = {1: np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]), 2: np.array([0.0, 0.0, 1.0, 0, 0, 0])}
    cand = {1: np.array([1.1, 2.0, 3.0, np.nan, np.nan, np.nan]), 2: np.array([0.0, 0.0, 1.0, np.nan, np.nan, np.nan])}
    stats = StressDistributionVerifier().compare_displacements(ref, cand, dofs=range(6))
    assert set(stats) == {"UX", "UY", "UZ"}
    assert stats["UX"].max_absolute_error == pytest.approx(0.1)
    table = StressDistributionVerifier.format_table(stats.values(), "t")
    assert "UX" in table and "MAE" in table


def test_benchmark_runner_completes_without_external_solvers(tmp_path, capsys):
    results = run_solver_benchmarks.main(
        ["--n", "4", "--work-dir", str(tmp_path), "--ccx", "no-such-ccx", "--nastran", "no-such-nastran"]
    )
    out = capsys.readouterr().out
    assert results["native"].status is KernelStatus.SUCCESS
    assert results["calculix"].status is KernelStatus.SKIPPED
    assert results["nastran"].status is KernelStatus.SKIPPED
    assert "Benchmark summary" in out and "max|UZ|" in out
    assert (tmp_path / "calculix" / "asymmetric_plate.inp").exists()
    assert (tmp_path / "nastran" / "asymmetric_plate.bdf").exists()
