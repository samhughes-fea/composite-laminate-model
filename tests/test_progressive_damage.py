import numpy as np
import pytest

from laminate_modelling_package import (
    DamageState,
    ElasticProperties3D,
    FailureMode,
    FailureResult,
    MaximumStressCriterion,
    ProgressiveDamageAnalysis,
    SmearedCrackDegradation,
    integrate_abd,
)
from test_stress_recovery import tension_strip

E = ElasticProperties3D.transversely_isotropic(181e9, 10.3e9, 0.28, 7.17e9, 0.4)


def test_matrix_cracking_preserves_fibre_stiffness():
    degraded = SmearedCrackDegradation(residual_fraction=0.05).degrade(E, DamageState(matrix_tension=True))
    assert degraded.E1 == E.E1 and degraded.G13 == E.G13
    assert degraded.E2 == pytest.approx(0.05 * E.E2)
    assert degraded.G12 == pytest.approx(0.05 * E.G12)
    assert degraded.G23 == pytest.approx(0.05 * E.G23)
    assert degraded.v12 == pytest.approx(0.05 * E.v12)


def test_fibre_failure_and_residual_floor():
    strategy = SmearedCrackDegradation(residual_fraction=0.0, minimum_fraction=1e-4)
    degraded = strategy.degrade(E, DamageState(fiber_tension=True))
    assert degraded.E1 == pytest.approx(1e-4 * E.E1)
    assert degraded.E2 == E.E2
    assert degraded.v12 == 0.0


def test_delamination_and_shear_flags():
    s = SmearedCrackDegradation(0.1)
    d = s.degrade(E, DamageState(delamination=True))
    assert d.E3 == pytest.approx(0.1 * E.E3) and d.G13 == pytest.approx(0.1 * E.G13) and d.E1 == E.E1
    sh = s.degrade(E, DamageState(shear=True))
    assert sh.G12 == pytest.approx(0.1 * E.G12) and sh.E2 == E.E2
    assert s.degrade(E, DamageState()) is E
    with pytest.raises(ValueError):
        SmearedCrackDegradation(1.5)


def test_damage_state_merging_and_mapping():
    a = DamageState(matrix_tension=True)
    b = DamageState(fiber_compression=True)
    merged = a.merge(b)
    assert merged.matrix and merged.fiber and merged.labels() == ["fiber_compression", "matrix_tension"]
    result = FailureResult("x", 1.2, 0.9, FailureMode.MATRIX_TENSION, {"matrix_tension": 1.2, "fiber_tension": 0.3})
    assert DamageState.from_failure(result) == DamageState(matrix_tension=True)
    interactive = FailureResult("tw", 1.1, 0.95, FailureMode.INTERACTIVE, {"fiber": 0.1, "matrix": 0.9, "shear": 0.1, "interlaminar": 0.0})
    assert DamageState.from_interactive(interactive, np.array([1, -5, 0, 0, 0, 0])) == DamageState(matrix_compression=True)


def first_ply_failure_load(lam):
    """Classical laminate theory: N_x at which the 90-degree plies reach Yt (max stress)."""
    eps0 = np.linalg.solve(integrate_abd(lam)[:3, :3], [1.0, 0.0, 0.0])
    ply90 = lam.plies[1]
    sigma2 = (ply90.T @ ply90.Q_bar @ eps0)[1]
    return ply90.material.strength.Yt / sigma2


@pytest.mark.parametrize("criterion", [None, MaximumStressCriterion()], ids=["Hashin3D", "MaxStress"])
def test_first_ply_failure_of_cross_ply_in_tension(cfrp, criterion):
    lam, _, _, _ = tension_strip(cfrp, [0, 90, 90, 0])
    n_fpf = first_ply_failure_load(lam)
    lam, mesh, bcs, loads = tension_strip(cfrp, [0, 90, 90, 0], force_per_width=n_fpf)
    analysis = ProgressiveDamageAnalysis(criterion=criterion, degradation=SmearedCrackDegradation(0.01))
    result = analysis.run(mesh, bcs, loads, load_factors=[0.5, 0.98, 1.02, 1.3])
    assert result.first_ply_failure_factor == pytest.approx(1.02)
    assert result.steps[1].newly_failed == 0
    assert result.steps[2].newly_failed == 2 * mesh.n_elements
    for (eid, ply), state in result.damage.items():
        assert ply in (1, 2)
        assert state.matrix_tension and not state.fiber
    for element in result.mesh.elements:
        assert element.laminate.plies[1].material.elastic.E2 == pytest.approx(0.01 * cfrp.elastic.E2)
        assert element.laminate.plies[0].material.elastic == cfrp.elastic
    # Original mesh is untouched.
    assert mesh.elements[0].laminate.plies[1].material is cfrp
    # Stiffness loss: displacement grows faster than load after the 90-degree plies crack.
    disp = [s.max_displacement for s in result.steps]
    assert disp[3] / disp[0] > 1.05 * (1.3 / 0.5)
    assert disp[1] / disp[0] == pytest.approx(0.98 / 0.5, rel=1e-9)
