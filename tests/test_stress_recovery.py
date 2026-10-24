import numpy as np
import pytest

from laminate_modelling_package import (
    DOF,
    BoundaryConditions,
    ExactStressRecoverer,
    FEAMesh,
    FSDTModel,
    Laminate,
    LoadCase,
    RefinedZigZagModel,
    StructuralGlobalSolver,
    integrate_abd,
)
from conftest import simply_supported


def tension_strip(cfrp, stack, force_per_width=1.0e5, n=4):
    lam = Laminate.from_angles(cfrp, stack, 0.25e-3)
    L, b = 0.1, 0.04
    mesh = FEAMesh.rectangular_plate(L, b, n, 2, lam, FSDTModel())
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    bcs.fix(mesh.nodes_on_line(0, 0.0), [DOF.UX])
    bcs.fix([mesh.closest_node((0.0, 0.0))], [DOF.UY])
    loads = LoadCase()
    right = sorted(mesh.nodes_on_line(0, L), key=lambda nid: mesh.node_coordinates(nid)[1])
    edge = b / 2
    for i, node in enumerate(right):
        share = 0.5 if i in (0, len(right) - 1) else 1.0
        loads.add_nodal_load(node, DOF.UX, force_per_width * edge * share)
    return lam, mesh, bcs, loads


def test_ply_stresses_match_classical_laminate_theory(cfrp):
    lam, mesh, bcs, loads = tension_strip(cfrp, [0, 90, 90, 0])
    result = StructuralGlobalSolver().solve(mesh, bcs, loads)
    eps0 = np.linalg.solve(integrate_abd(lam)[:3, :3], [1.0e5, 0.0, 0.0])
    element = mesh.elements[3]
    profile = ExactStressRecoverer().through_thickness_profile(element, result.element_displacements(element.element_id))
    for point in profile:
        ply = lam.plies[point.ply_index]
        assert np.allclose(point.stress_global, ply.Q_bar @ eps0, rtol=1e-8, atol=1.0)
        assert np.allclose(point.stress_material, ply.T @ (ply.Q_bar @ eps0), rtol=1e-8, atol=1.0)
    assert len(profile) == 3 * lam.n_plies


def test_off_axis_material_transformation(cfrp):
    lam, mesh, bcs, loads = tension_strip(cfrp, [45, -45, -45, 45])
    result = StructuralGlobalSolver().solve(mesh, bcs, loads)
    element = mesh.elements[2]
    point = ExactStressRecoverer().recover(element, result.element_displacements(element.element_id), 0.4e-3)
    ply = lam.plies[point.ply_index]
    s1, s2, t12 = point.stress_material
    sx, sy, txy = point.stress_global
    c = s = np.sqrt(0.5)
    assert s1 == pytest.approx(c * c * sx + s * s * sy + 2 * c * s * txy, rel=1e-10)
    assert point.material_stress_3d()[[0, 1, 5]] == pytest.approx([s1, s2, t12])


def test_condensed_rzt_transverse_shear_is_continuous_in_fe_solution(cfrp):
    """With the zig-zag amplitude slaved to gamma0, tau = G gamma0 in every ply."""
    lam = Laminate.from_angles(cfrp, [0, 90, 0], 1.0e-3)
    mesh = FEAMesh.rectangular_plate(0.06, 0.06, 6, 6, lam, RefinedZigZagModel(condensed=True))
    result = StructuralGlobalSolver().solve(mesh, simply_supported(mesh, 0.06, 0.06), LoadCase().add_pressure(1e5))
    element = mesh.elements[1]
    u = result.element_displacements(element.element_id)
    rec = ExactStressRecoverer()
    for z in lam.z_interfaces[1:-1]:
        below = rec.recover(element, u, z, side="below").shear_global
        above = rec.recover(element, u, z, side="above").shear_global
        assert np.allclose(below, above, rtol=1e-10)
        assert np.linalg.norm(above) > 0.0


def test_sampling_points_cover_each_ply(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90], 1e-3)
    mesh = FEAMesh.rectangular_plate(0.1, 0.1, 1, 1, lam, FSDTModel())
    samples = ExactStressRecoverer().ply_sampling_points(mesh.elements[0], 3)
    assert [s[0] for s in samples] == pytest.approx([-1e-3, -0.5e-3, 0.0, 0.0, 0.5e-3, 1e-3])
    assert [s[1] for s in samples] == ["above", "above", "below", "above", "above", "below"]
    with pytest.raises(ValueError):
        ExactStressRecoverer().ply_sampling_points(mesh.elements[0], 0)
