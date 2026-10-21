import numpy as np
import pytest

from conftest import navier_clpt_centre, navier_fsdt_centre, simply_supported
from laminate_modelling_package import (
    DOF,
    BoundaryConditions,
    CLPTModel,
    DenseLinearSolver,
    FEAMesh,
    FSDTModel,
    GlobalAssembler,
    Laminate,
    LayerwiseModel,
    LoadCase,
    RefinedZigZagModel,
    SingularStiffnessError,
    StructuralGlobalSolver,
    integrate_abd,
)

A = 0.2
Q = 1.0e4


def solve_ss_plate(cfrp, a_over_h, kinematics, n=16):
    h = A / a_over_h
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], h / 4)
    mesh = FEAMesh.rectangular_plate(A, A, n, n, lam, kinematics)
    result = StructuralGlobalSolver().solve(mesh, simply_supported(mesh, A, A), LoadCase().add_pressure(Q))
    return lam, mesh, result, result.node_displacement(mesh.closest_node((A / 2, A / 2)))[2]


@pytest.mark.parametrize("kinematics", [CLPTModel(), FSDTModel(), RefinedZigZagModel(), LayerwiseModel()])
def test_thin_plate_matches_clpt_navier_solution(cfrp, kinematics):
    lam, _, _, w = solve_ss_plate(cfrp, 100, kinematics)
    reference = navier_clpt_centre(integrate_abd(lam)[3:, 3:], A, A, Q)
    assert w == pytest.approx(reference, rel=0.01)


def test_thick_plate_clpt_model_stays_shear_rigid(cfrp):
    lam, _, _, w = solve_ss_plate(cfrp, 10, CLPTModel())
    assert w == pytest.approx(navier_clpt_centre(integrate_abd(lam)[3:, 3:], A, A, Q), rel=0.01)


def test_thick_plate_fsdt_matches_fsdt_navier_solution(cfrp):
    lam, _, _, w = solve_ss_plate(cfrp, 10, FSDTModel())
    fsdt = FSDTModel()
    reference = navier_fsdt_centre(integrate_abd(lam)[3:, 3:], fsdt.transverse_shear_matrix(lam), A, A, Q)
    assert w == pytest.approx(reference, rel=0.01)
    # Shear deformation makes the thick plate noticeably softer than Kirchhoff theory.
    assert w > 1.15 * navier_clpt_centre(integrate_abd(lam)[3:, 3:], A, A, Q)


def test_reactions_balance_applied_pressure(cfrp):
    _, mesh, result, _ = solve_ss_plate(cfrp, 50, FSDTModel(), n=8)
    assert np.sum(result.reactions[2::6]) == pytest.approx(-Q * A * A, rel=1e-9)


def membrane_strip(cfrp, n=6):
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.25e-3)
    L, b = 0.1, 0.02
    mesh = FEAMesh.rectangular_plate(L, b, n, 2, lam, FSDTModel())
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    bcs.fix(mesh.nodes_on_line(0, 0.0), [DOF.UX])
    bcs.fix([mesh.closest_node((0.0, 0.0))], [DOF.UY])
    return lam, mesh, bcs, L, b


def test_prescribed_displacement_and_reaction(cfrp):
    lam, mesh, bcs, L, b = membrane_strip(cfrp)
    delta = 1e-5
    bcs.fix(mesh.nodes_on_line(0, L), [DOF.UX], value=delta)
    result = StructuralGlobalSolver().solve(mesh, bcs, LoadCase())
    a11 = np.linalg.inv(integrate_abd(lam)[:3, :3])[0, 0]
    expected_force = b * delta / L / a11
    reaction = sum(result.node_reaction(n)[0] for n in mesh.nodes_on_line(0, L))
    assert reaction == pytest.approx(expected_force, rel=1e-8)
    assert result.node_displacement(mesh.closest_node((L / 2, 0.0)))[0] == pytest.approx(delta / 2, rel=1e-8)


def test_multi_point_constraints(cfrp):
    lam, mesh, bcs, L, b = membrane_strip(cfrp)
    right = sorted(mesh.nodes_on_line(0, L))
    master = right[0]
    for node in right[1:]:
        bcs.tie((node, DOF.UX), (master, DOF.UX))
    loads = LoadCase().add_nodal_load(master, DOF.UX, 100.0)
    result = StructuralGlobalSolver().solve(mesh, bcs, loads)
    ux = [result.node_displacement(n)[0] for n in right]
    assert np.allclose(ux, ux[0])
    a11 = np.linalg.inv(integrate_abd(lam)[:3, :3])[0, 0]
    assert ux[0] == pytest.approx(100.0 / b * a11 * L, rel=1e-6)


def test_constraint_with_constant_offset(cfrp):
    lam, mesh, bcs, L, b = membrane_strip(cfrp)
    n1, n2 = mesh.closest_node((L, 0.0)), mesh.closest_node((L, b))
    bcs.fix([n1], [DOF.UX], value=2e-5)
    bcs.add_constraint((n2, DOF.UX), [((n1, DOF.UX), 1.0)], constant=-1e-5)
    result = StructuralGlobalSolver().solve(mesh, bcs, LoadCase())
    assert result.node_displacement(n2)[0] == pytest.approx(1e-5)


def test_invalid_constraints_are_rejected(cfrp):
    lam, mesh, bcs, L, b = membrane_strip(cfrp)
    n = mesh.closest_node((L, b))
    bcs.fix([n], [DOF.UX])
    bcs.tie((n, DOF.UX), (1, DOF.UY))
    with pytest.raises(ValueError):
        StructuralGlobalSolver().solve(mesh, bcs, LoadCase())


def test_unconstrained_model_is_singular(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90], 0.25e-3)
    mesh = FEAMesh.rectangular_plate(0.1, 0.1, 2, 2, lam, FSDTModel())
    with pytest.raises(SingularStiffnessError):
        StructuralGlobalSolver().solve(mesh, BoundaryConditions(), LoadCase().add_pressure(1.0))
    with pytest.raises(SingularStiffnessError):
        StructuralGlobalSolver(GlobalAssembler(sparse=False), DenseLinearSolver()).solve(
            mesh, BoundaryConditions(), LoadCase().add_pressure(1.0)
        )


def test_dense_and_sparse_paths_agree(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 45, -45, 90], 0.25e-3)
    mesh = FEAMesh.rectangular_plate(0.1, 0.1, 4, 4, lam, FSDTModel())
    bcs = simply_supported(mesh, 0.1, 0.1, hard=False)
    loads = LoadCase().add_pressure(500.0)
    sparse = StructuralGlobalSolver().solve(mesh, bcs, loads)
    dense = StructuralGlobalSolver(GlobalAssembler(sparse=False), DenseLinearSolver()).solve(mesh, bcs, loads)
    assert np.allclose(sparse.displacements, dense.displacements, rtol=1e-9, atol=1e-11 * np.abs(dense.displacements).max())


def test_mesh_factory_and_dof_map(cfrp):
    lam = Laminate.from_angles(cfrp, [0], 1e-3)
    mesh = FEAMesh.rectangular_plate(1.0, 0.5, 3, 2, lam, FSDTModel())
    assert mesh.n_nodes == 12 and mesh.n_elements == 6 and mesh.n_dof == 72
    assert list(mesh.element_dofs(1)[:6]) == list(range(6))
    assert mesh.closest_node((1.0, 0.5)) == 12
    with pytest.raises(ValueError):
        mesh.add_node(1, (0, 0, 0))
    with pytest.raises(KeyError):
        mesh.add_element(99, (1, 2, 3, 999), lam, FSDTModel())
