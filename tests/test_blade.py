"""Blade builder, connections (RBE2 / RBE3) and beam section properties."""

import numpy as np
import pytest

from composite_laminate_model import (
    DOF,
    Airfoil,
    BladeBuilder,
    BoundaryConditions,
    CalculiXDeckWriter,
    AnalysisModel,
    CylindricalPanelBuilder,
    FSDTModel,
    Laminate,
    Layup,
    LoadCase,
    Ply,
    PlyDefinition,
    ShearWeb,
    SkinRegion,
    Station,
    StructuralGlobalSolver,
    add_reference_node,
    beam_section_properties,
    distributing_link,
    rigid_link,
)

REGIONS = [SkinRegion("le_panel", 0.0, 0.15, 3), SkinRegion("spar_cap", 0.15, 0.45, 4),
           SkinRegion("te_panel", 0.45, 1.0, 5)]
WEBS = [ShearWeb("web_fwd", 0.15, 3), ShearWeb("web_aft", 0.45, 3)]


def laminates(cfrp):
    skin = Laminate.from_angles(cfrp, [45, -45, 0, -45, 45], 0.25e-3)
    return {
        "le_panel": skin, "te_panel": skin,
        "spar_cap": Laminate.from_angles(cfrp, [45, -45] + [0] * 8 + [-45, 45], 0.25e-3),
        "web_fwd": Laminate.from_angles(cfrp, [45, -45, -45, 45], 0.25e-3),
        "web_aft": Laminate.from_angles(cfrp, [45, -45, -45, 45], 0.25e-3),
    }


def test_airfoils():
    naca = Airfoil.naca4("0018")
    assert naca.max_thickness == pytest.approx(0.18, rel=2e-3)
    assert naca.surface("upper", 0.3) == pytest.approx(-naca.surface("lower", 0.3))
    cambered = Airfoil.naca4("4412")
    assert cambered.surface("upper", 0.4) + cambered.surface("lower", 0.4) == pytest.approx(0.08, abs=2e-3)
    circle = Airfoil.ellipse(1.0)
    assert circle.thickness(0.5) == pytest.approx(1.0, rel=1e-6)
    # Selig order or reversed gives the same surfaces
    pts = np.array([[1, 0], [0.5, 0.1], [0, 0], [0.5, -0.05], [1, 0]], float)
    assert Airfoil(pts).surface("upper", 0.5) == pytest.approx(Airfoil(pts[::-1]).surface("upper", 0.5))
    with pytest.raises(ValueError):
        Airfoil.naca4("18")


@pytest.mark.parametrize("element", ["quad4", "quad8", "quad8_curved"])
def test_blade_builder_regions_webs_and_normals(cfrp, element):
    af = Airfoil.naca4("0021")
    stations = [Station(0.0, 1.0, af, twist=0.0, pitch_axis=0.3), Station(3.0, 0.7, Airfoil.naca4("0015"), twist=-6.0,
                                                                           pitch_axis=0.3)]
    blade = BladeBuilder(stations, REGIONS, WEBS, n_span=6, element=element, mesh_surface="outer").build(
        laminates(cfrp), FSDTModel())
    mesh = blade.mesh
    sets = blade.element_sets
    for name in ("le_panel_ss", "le_panel_ps", "spar_cap_ss", "spar_cap_ps", "te_panel", "web_fwd", "web_aft",
                 "skin", "webs"):
        assert sets[name], name
    assert len(sets["spar_cap"]) == 2 * 4 * 6 and len(sets["web_fwd"]) == 3 * 6
    assert sorted(sets["all"]) == sorted(sets["skin"] + sets["webs"])
    assert {e.TOPOLOGY for e in mesh.elements} == {element}
    # skin normals point out of the section, web normals towards the trailing edge
    for eid in sets["skin"]:
        e = mesh.element(eid)
        assert e.frame[2][:2] @ (np.asarray(e.centroid)[:2] - _section_centre(blade, e.centroid[2])) > 0
    for eid in sets["web_aft"]:
        e = mesh.element(eid)
        a, b, w, chord, twist, *_ = BladeBuilder(stations, REGIONS, WEBS)._station_at(e.centroid[2])
        chord_dir = np.array([np.cos(np.radians(twist)), -np.sin(np.radians(twist))])
        assert e.frame[2][:2] @ chord_dir > 0.9
    # outer mould line reference: the laminate grows inward
    cap = mesh.element(sets["spar_cap_ss"][0])
    assert cap.laminate.z_interfaces[-1] == pytest.approx(0.0, abs=1e-15)
    assert abs(cap.frame[0][2]) > 0.99  # 0-degree plies run along the span
    # webs share nodes with the skins
    joint = set(blade.nodes("web_fwd_ss"))
    skin_nodes = {n for eid in sets["spar_cap_ss"] for n in mesh.element(eid).node_ids}
    assert joint <= skin_nodes
    assert len(blade.nodes("root")) == len(blade.nodes("tip"))
    # tip of the twisted, tapered blade: chord 0.7 rotated by -6 degrees
    le = [n for n in blade.nodes("le") if n in blade.nodes("tip")][0]
    x = mesh.node_coordinates(le)
    assert x[0] == pytest.approx(-0.3 * 0.7 * np.cos(np.radians(6)), abs=1e-4)
    assert x[1] == pytest.approx(-0.3 * 0.7 * np.sin(np.radians(6)), abs=1e-4)


def _section_centre(blade, z):
    pts = np.array([blade.mesh.node_coordinates(n)[:2] for n in blade.node_sets["all"]
                    if abs(blade.mesh.node_coordinates(n)[2] - z) < 0.3])
    return pts.mean(axis=0)


def test_blunt_trailing_edges_get_a_web_and_errors_are_clear(cfrp):
    pts = Airfoil.naca4("0018", closed_te=False)
    blade = BladeBuilder([Station(0, 1, pts), Station(1, 1, pts)], REGIONS, n_span=2).build(
        laminates(cfrp) | {"te_web": laminates(cfrp)["web_fwd"]}, FSDTModel())
    assert blade.element_sets["te_web"]
    with pytest.raises(ValueError, match="region boundary"):
        BladeBuilder([Station(0, 1, pts), Station(1, 1, pts)], REGIONS, [ShearWeb("w", 0.3)])
    with pytest.raises(ValueError, match="gap"):
        BladeBuilder([Station(0, 1, pts), Station(1, 1, pts)], [SkinRegion("a", 0, 0.4), SkinRegion("b", 0.5, 1)])
    with pytest.raises(KeyError, match="No laminate"):
        BladeBuilder([Station(0, 1, pts), Station(1, 1, pts)], REGIONS, n_span=1).build({}, FSDTModel())


@pytest.mark.parametrize("element", ["quad4", "quad8_curved"])
def test_beam_properties_of_a_thin_tube(aluminium, element):
    t, R = 2e-3, 0.25
    lam = Laminate([Ply(aluminium, t)])
    circle = Airfoil.ellipse(1.0, 161)
    blade = BladeBuilder([Station(0, 2 * R, circle, pitch_axis=0.5), Station(6, 2 * R, circle, pitch_axis=0.5)],
                         [SkinRegion("a", 0, 0.5, 8), SkinRegion("b", 0.5, 1.0, 8)], n_span=24, element=element,
                         skin_reference="mid").build(lam, FSDTModel())
    p = beam_section_properties(blade)
    E = aluminium.elastic.E1
    G = E / (2 * (1 + aluminium.elastic.v12))
    tol = 0.012 if element == "quad4" else 1e-3
    assert p.EA == pytest.approx(E * 2 * np.pi * R * t, rel=tol)
    assert p.EI_x == pytest.approx(E * np.pi * R**3 * t, rel=tol)
    assert p.EI_y == pytest.approx(E * np.pi * R**3 * t, rel=tol)
    assert p.GJ == pytest.approx(G * 2 * np.pi * R**3 * t, rel=tol)
    assert np.allclose(p.shear_centre, 0.0, atol=1e-6) and np.allclose(p.tension_centre, 0.0, atol=1e-6)
    assert p.mass_per_length == pytest.approx(aluminium.density * 2 * np.pi * R * t, rel=tol)
    assert np.allclose(p.bend_twist_coupling, 0.0, atol=1e-6)


def _thick_tube_gj_ratio(aluminium, element, reference, t_over_r=0.04, mesh_surface="outer"):
    """GJ of a thick tube (outer radius 1) over the exact G 2 pi R^3 t, R the mid-wall radius."""
    R, t = 1.0, t_over_r
    circle = Airfoil.ellipse(1.0)
    n = 24 if element == "quad4" else 12
    blade = BladeBuilder([Station(0, 2 * R, circle, pitch_axis=0.5), Station(8, 2 * R, circle, pitch_axis=0.5)],
                         [SkinRegion("a", 0, 0.5, n), SkinRegion("b", 0.5, 1.0, n)], n_span=16, element=element,
                         skin_reference=reference, mesh_surface=mesh_surface).build(Laminate([Ply(aluminium, t)]),
                                                                                    FSDTModel())
    G = aluminium.elastic.E1 / (2 * (1 + aluminium.elastic.v12))
    r_mid = R - 0.5 * t if reference == "outer" else R
    return beam_section_properties(blade).GJ / (G * 2 * np.pi * r_mid**3 * t)


@pytest.mark.parametrize("element", ["quad4", "quad8", "quad8_curved"])
def test_mid_surface_mesh_keeps_the_torsion_of_a_thick_tube(aluminium, element):
    assert _thick_tube_gj_ratio(aluminium, element, "outer", 0.06, "mid") == pytest.approx(1.0, abs=0.005)


def test_mid_surface_mesh_moves_skins_by_their_offsets_and_reattaches_webs(cfrp):
    thin = Laminate.from_angles(cfrp, [45, -45, 0, -45, 45], 0.25e-3)        # 1.25 mm
    cap = Laminate.from_angles(cfrp, [45, -45] + [0] * 20 + [-45, 45], 0.25e-3)  # 6 mm
    lams = {"le_panel": thin, "te_panel": thin, "spar_cap": cap, "web_fwd": thin, "web_aft": thin}
    stations = [Station(0, 1.0, Airfoil.naca4("0024"), pitch_axis=0.25), Station(2, 1.0, Airfoil.naca4("0024"),
                                                                                  pitch_axis=0.25)]
    outer = BladeBuilder(stations, REGIONS, WEBS, n_span=4, element="quad4", mesh_surface="outer").build(
        lams, FSDTModel())
    mid = BladeBuilder(stations, REGIONS, WEBS, n_span=4, element="quad4",
                       mid_surface_smoothing=0.02).build(lams, FSDTModel())  # mid-surface is the default
    assert outer.surface == "outer" and mid.surface == "mid"
    assert mid.mesh.node_ids == outer.mesh.node_ids and mid.element_sets == outer.element_sets
    # every ply stays where it was: through the thickness, the element's mid-plane is
    # the old reference surface plus the old offset
    for eid in outer.elements("skin"):
        e0, e1 = outer.mesh.element(eid), mid.mesh.element(eid)
        n0 = np.asarray(e0.frame)[2]
        p0 = np.asarray(e0.centroid) + e0.laminate.offset * n0
        p1 = np.asarray(e1.centroid) + e1.laminate.offset * np.asarray(e1.frame)[2]
        assert abs(np.dot(p1 - p0, n0)) < 0.1 * e0.laminate.thickness
    # away from the cap edges (and the sharp trailing edge) the skins keep almost no offset
    small = [abs(mid.mesh.element(e).laminate.offset) < 0.05 * mid.mesh.element(e).laminate.thickness
             for e in mid.elements("skin")]
    assert sum(small) > 0.5 * len(small)
    # sections stay in their planes, and the webs end on the moved skin nodes
    for n in mid.nodes("root"):
        assert mid.mesh.node_coordinates(n)[2] == pytest.approx(0.0, abs=1e-12)
    for n in mid.nodes("web_fwd_ss"):  # on a cap edge: between the two half-thicknesses
        move = np.linalg.norm(mid.mesh.node_coordinates(n) - outer.mesh.node_coordinates(n))
        assert 0.5 * thin.thickness < move < 0.5 * cap.thickness


def test_mid_surface_after_a_layup_follows_the_ply_drops(cfrp):
    stations = [Station(0, 1.0, Airfoil.naca4("0024"), pitch_axis=0.25), Station(2, 1.0, Airfoil.naca4("0024"),
                                                                                  pitch_axis=0.25)]
    plies = [PlyDefinition(f"ud{k}", cfrp, 1e-3, 0, elements="spar_cap",
                           region=lambda x, k=k: x[2] < 2.0 - 0.4 * k) for k in range(4)]
    # the ply book refers to the mould line, as laid in the mould
    layup = Layup(plies, reference="top")
    outer = BladeBuilder(stations, REGIONS, WEBS, n_span=8, element="quad4", mesh_surface="outer").build(
        laminates(cfrp), FSDTModel())
    layup.apply(outer.mesh, outer.element_sets, element_ids=outer.elements("spar_cap"))
    mid = outer.to_mid_surface(smoothing=0.05)
    node = lambda z: min(mid.nodes("web_fwd_ss"), key=lambda n: abs(outer.mesh.node_coordinates(n)[2] - z))  # noqa
    move = lambda z: np.linalg.norm(mid.mesh.node_coordinates(node(z)) - outer.mesh.node_coordinates(node(z)))  # noqa
    assert move(0.1) > move(1.9)  # the cap is thicker at the root, so its mid-surface sits deeper

    # the same ply book on a blade built on its mid-surface (the default): moving it
    # again from the mould line gives the same mesh
    blade = BladeBuilder(stations, REGIONS, WEBS, n_span=8, element="quad4",
                         mid_surface_smoothing=0.05).build(laminates(cfrp), FSDTModel())
    layup.apply(blade.mesh, blade.element_sets, element_ids=blade.elements("spar_cap"))
    again = blade.to_mid_surface()
    assert again.surface == "mid"
    for n in mid.mesh.node_ids:
        assert again.mesh.node_coordinates(n) == pytest.approx(mid.mesh.node_coordinates(n), abs=1e-12)
    for e in mid.mesh.elements:
        assert again.mesh.element(e.element_id).laminate.offset == pytest.approx(e.laminate.offset, abs=1e-12)
    # and redoing it with no new plies changes nothing
    twice = again.to_mid_surface()
    for n in mid.mesh.node_ids:
        assert twice.mesh.node_coordinates(n) == pytest.approx(mid.mesh.node_coordinates(n), abs=1e-12)


def test_curved_shells_with_offset_skins_keep_the_torsion_of_a_thick_tube(aluminium):
    # The 2% left over is the offset skin's area, taken on the outer surface (EA is 2% high too).
    assert _thick_tube_gj_ratio(aluminium, "quad8_curved", "outer") == pytest.approx(1.0, abs=0.03)
    assert _thick_tube_gj_ratio(aluminium, "quad4", "mid") == pytest.approx(1.0, abs=0.01)


@pytest.mark.xfail(strict=True, reason="known limitation: flat shells (Quad4, flat Quad8) with skins offset from "
                                       "the reference surface lose torsional stiffness in thick closed sections "
                                       "(about 22% at t/R = 0.04, 35% at 0.06); use quad8_curved")
def test_flat_shells_with_offset_skins_keep_the_torsion_of_a_thick_tube(aluminium):
    assert _thick_tube_gj_ratio(aluminium, "quad4", "outer") == pytest.approx(1.0, abs=0.03)


def test_beam_properties_show_bend_twist_coupling_and_offset_centres(cfrp):
    af = Airfoil.naca4("0024")
    lams = laminates(cfrp)
    stations = [Station(0, 1.0, af, pitch_axis=0.25), Station(6, 1.0, af, pitch_axis=0.25)]
    blade = BladeBuilder(stations, REGIONS, WEBS, n_span=24).build(lams, FSDTModel())
    plain = beam_section_properties(blade)
    assert max(np.abs(plain.bend_twist_coupling)) < 5e-3
    # box spar between the webs: the shear centre lies aft of the quarter chord, inside the box
    assert 0.0 < plain.shear_centre[0] < 0.2 and abs(plain.shear_centre[1]) < 1e-3
    assert 0.0 < plain.tension_centre[0] < 0.2
    # off-axis cap fibres: opposite angles on the two sides couple flap bending and twist,
    # equal angles (mirrored by the opposite normals) couple extension and twist
    off = lambda a: Laminate.from_angles(cfrp, [a] * 12, 0.25e-3)  # noqa: E731
    bend_twist = beam_section_properties(BladeBuilder(stations, REGIONS, WEBS, n_span=24).build(
        dict(lams, spar_cap_ss=off(20), spar_cap_ps=off(-20)), FSDTModel()))
    ext_twist = beam_section_properties(BladeBuilder(stations, REGIONS, WEBS, n_span=24).build(
        dict(lams, spar_cap=off(20)), FSDTModel()))
    S = bend_twist.stiffness
    assert abs(bend_twist.bend_twist_coupling[0]) > 0.2
    assert abs(S[0, 3]) / np.sqrt(S[0, 0] * S[3, 3]) < 0.05
    S = ext_twist.stiffness
    assert abs(S[0, 3]) / np.sqrt(S[0, 0] * S[3, 3]) > 0.1 and abs(ext_twist.bend_twist_coupling[0]) < 0.05


def test_rigid_and_distributing_links(aluminium):
    lam = Laminate([Ply(aluminium, 2e-3)])
    R, L = 0.25, 3.0
    named = CylindricalPanelBuilder(R, L, 12, 16, 0, 360, element="quad8_curved").build_named(lam, FSDTModel())
    mesh = named.mesh
    bcs = BoundaryConditions().clamp(named.nodes("x0"))
    ref = add_reference_node(mesh, (L, 0.0, 0.0))
    distributing_link(mesh, bcs, ref, named.nodes("x1"))
    E = aluminium.elastic.E1
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_nodal_load(ref, DOF.UZ, 100.0))
    # Euler-Bernoulli plus a small Timoshenko shear term for a thin tube
    I, A = np.pi * R**3 * 2e-3, 2 * np.pi * R * 2e-3
    G = E / (2 * (1 + aluminium.elastic.v12))
    w = 100.0 * L**3 / (3 * E * I) + 100.0 * L / (0.5 * G * A)
    assert sol.node_displacement(ref)[2] == pytest.approx(w, rel=0.02)
    # RBE2: a rigid end keeps the section plane and round
    bcs2 = BoundaryConditions().clamp(named.nodes("x0"))
    ref2 = add_reference_node(mesh, (L, 0.0, 0.0))
    rigid_link(mesh, bcs2, ref2, named.nodes("x1"))
    bcs2.fix([ref], range(6))  # the unused RBE3 node from above
    sol2 = StructuralGlobalSolver().solve(mesh, bcs2, LoadCase().add_nodal_load(ref2, DOF.RX, 50.0))
    twist = sol2.node_displacement(ref2)[3]
    assert twist == pytest.approx(50.0 * L / (G * 2 * np.pi * R**3 * 2e-3), rel=0.01)
    for n in named.nodes("x1"):
        x = mesh.node_coordinates(n) - mesh.node_coordinates(ref2)
        assert np.allclose(sol2.node_displacement(n)[:3], np.cross([twist, 0, 0], x), atol=1e-12)
    with pytest.raises(ValueError, match="collinear"):
        distributing_link(mesh, BoundaryConditions(), ref, named.nodes("x1")[:1] * 3)


def test_links_are_written_to_decks(aluminium, tmp_path):
    lam = Laminate([Ply(aluminium, 2e-3)])
    named = CylindricalPanelBuilder(0.25, 1.0, 4, 8, 0, 360).build_named(lam, FSDTModel())
    bcs = BoundaryConditions().clamp(named.nodes("x0"))
    ref = add_reference_node(named.mesh, (1.0, 0.0, 0.0))
    distributing_link(named.mesh, bcs, ref, named.nodes("x1"))
    deck = CalculiXDeckWriter().write(AnalysisModel(named.mesh, bcs, LoadCase().add_nodal_load(ref, 2, 1.0)),
                                      tmp_path, "job").path.read_text()
    assert deck.count("*EQUATION") == 6
