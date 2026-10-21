"""Gmsh meshing: import, physical groups, orientation and an open-hole stress concentration."""

import numpy as np
import pytest

pytest.importorskip("gmsh", exc_type=ImportError)

from laminate_modelling_package import (  # noqa: E402
    DOF,
    BoundaryConditions,
    CircularHole,
    FSDTModel,
    GmshMeshImporter,
    GmshPlateBuilder,
    GmshSession,
    Laminate,
    LoadCase,
    ResultsExtractor,
    StructuralGlobalSolver,
    UnsupportedMeshError,
)


def tension(gm, length, nx=1.0e5):
    """Left edge held in x, one node held in y, line load nx on the right edge."""
    mesh = gm.mesh
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    bcs.fix(gm.nodes("left"), [DOF.UX]).fix([gm.nodes("left")[0]], [DOF.UY])
    right = sorted(gm.nodes("right"), key=lambda n: mesh.node_coordinates(n)[1])
    y = np.array([mesh.node_coordinates(n)[1] for n in right])
    tributary = np.zeros_like(y)
    tributary[:-1] += np.diff(y) / 2
    tributary[1:] += np.diff(y) / 2
    loads = LoadCase()
    for node, share in zip(right, tributary):
        loads.add_nodal_load(node, DOF.UX, nx * share)
    return StructuralGlobalSolver().solve(mesh, bcs, loads)


def test_plain_plate_is_all_quads_with_named_edges(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.25e-3)
    gm = GmshPlateBuilder(0.1, 0.05, size=0.01).build(GmshMeshImporter(lam, FSDTModel()))
    mesh = gm.mesh
    assert {"left", "right", "bottom", "top", "plate"} <= set(gm.node_sets)
    assert len(gm.elements("plate")) == len(mesh.elements) > 0
    for element in mesh.elements:
        assert element.frame[2] @ [0.0, 0.0, 1.0] > 0.999  # every normal faces +z
    xs = [mesh.node_coordinates(n)[0] for n in gm.nodes("right")]
    assert np.allclose(xs, 0.1)
    # Uniform tension gives a uniform membrane resultant everywhere.
    elements = ResultsExtractor(tension(gm, 0.1)).elements()
    assert np.allclose(elements.N[:, 0], 1.0e5, rtol=1e-6)
    assert np.allclose(elements.N[:, 1:], 0.0, atol=1e-3)


def test_open_hole_matches_isotropic_stress_concentration(aluminium):
    length, width, radius = 0.4, 0.2, 0.005
    lam = Laminate.from_angles(aluminium, [0], 2e-3)
    gm = GmshPlateBuilder(length, width, [CircularHole(length / 2, width / 2, radius)],
                          size=0.012, hole_size=0.0008, transition=6).build(GmshMeshImporter(lam, FSDTModel()))
    assert len(gm.nodes("hole_1")) > 40
    hole_radii = [np.hypot(*(gm.mesh.node_coordinates(n)[:2] - [length / 2, width / 2])) for n in gm.nodes("hole_1")]
    assert np.allclose(hole_radii, radius, rtol=1e-6)
    nodes = ResultsExtractor(tension(gm, length)).nodes()
    top = gm.mesh.closest_node((length / 2, width / 2 + radius))
    kt = nodes.N[list(nodes.node_ids).index(top), 0] / 1.0e5
    ratio = 1 - 2 * radius / width
    expected = (2 + ratio**3) / ratio  # Heywood finite-width formula, gross section
    assert kt == pytest.approx(expected, rel=0.03)


def test_physical_surfaces_select_laminates_and_msh_round_trip(cfrp, tmp_path):
    thin = Laminate.from_angles(cfrp, [0, 90], 0.25e-3)
    thick = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.25e-3)
    importer = GmshMeshImporter({"skin": thin, "doubler": thick}, FSDTModel())
    with GmshSession() as gmsh:
        gmsh.clear()
        a = gmsh.model.occ.addRectangle(0, 0, 0, 0.1, 0.05)
        b = gmsh.model.occ.addRectangle(0.1, 0, 0, 0.05, 0.05)
        gmsh.model.occ.fragment([(2, a)], [(2, b)])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [a], name="skin")
        gmsh.model.addPhysicalGroup(2, [b], name="doubler")
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)
        gmsh.model.mesh.generate(2)
        gm = importer.from_model()
        gmsh.write(str(tmp_path / "two_regions.msh"))
    skin, doubler = gm.elements("skin"), gm.elements("doubler")
    assert skin and doubler and not set(skin) & set(doubler)
    assert all(gm.mesh.element(e).laminate is thin for e in skin)
    assert all(gm.mesh.element(e).laminate is thick for e in doubler)
    # Shared interface nodes belong to both node sets, so the regions are connected.
    assert set(gm.nodes("skin")) & set(gm.nodes("doubler"))

    reread = importer.from_file(tmp_path / "two_regions.msh")
    assert reread.mesh.n_nodes == gm.mesh.n_nodes and len(reread.mesh.elements) == len(gm.mesh.elements)


def test_triangles_are_rejected_and_unmapped_regions_need_a_default(cfrp):
    lam = Laminate.from_angles(cfrp, [0], 0.25e-3)
    with GmshSession() as gmsh:
        gmsh.clear()
        s = gmsh.model.occ.addRectangle(0, 0, 0, 0.1, 0.1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [s], name="panel")
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)
        gmsh.model.mesh.generate(2)
        with pytest.raises(UnsupportedMeshError, match="RecombineAll"):
            GmshMeshImporter(lam, FSDTModel()).from_model()
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.clear()
        gmsh.model.mesh.generate(2)
        with pytest.raises(ValueError, match="default_laminate"):
            GmshMeshImporter({"other": lam}, FSDTModel()).from_model()
        flipped = GmshMeshImporter(lam, FSDTModel(), reference_normal=(0, 0, -1)).from_model()
    assert all(e.frame[2][2] < -0.999 for e in flipped.mesh.elements)


def test_remove_nodes_keeps_the_dof_map_consistent(cfrp):
    from laminate_modelling_package import FEAMesh

    lam = Laminate.from_angles(cfrp, [0], 0.25e-3)
    mesh = FEAMesh.rectangular_plate(0.1, 0.1, 1, 1, lam, FSDTModel())
    mesh.add_node(99, (0.5, 0.5, 0.0))
    with pytest.raises(ValueError):
        mesh.remove_nodes([1])
    mesh.remove_nodes([99])
    assert mesh.n_nodes == 4
    assert sorted(mesh.dof_index(n, 0) for n in mesh.node_ids) == [0, 6, 12, 18]
