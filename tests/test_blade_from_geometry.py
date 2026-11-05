"""BladeBuilder.from_geometry: shell models meshed from the implicit geometry."""

import numpy as np
import pytest

from composite_laminate_model import BladeBuilder, FSDTModel, Laminate, create_default_library
from implicit_geometry_kernel import Airfoil, BladeGeometry, Station


AIRFOIL = Airfoil.naca4("0024")


def _prismatic(twist=0.0, offset=(0.0, 0.0)):
    return [Station(0.0, 1.0, AIRFOIL, twist=twist, pitch_axis=0.3, offset=offset),
            Station(4.0, 1.0, AIRFOIL, twist=twist, pitch_axis=0.3, offset=offset)]


@pytest.fixture(scope="module")
def laminates():
    lib = create_default_library()
    glass, carbon = lib.get("E-glass/epoxy"), lib.get("T300/5208")
    return {
        "skin": Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3),
        "web": Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3),
        "cap": Laminate.from_angles(carbon, [0] * 16, 0.5e-3),
    }


def _geometry(stations=None, **kw):
    options = dict(spar_cap_thickness=0.008, shear_web_thickness=0.002, web_positions=(0.18, 0.48))
    options.update(kw)
    return BladeGeometry(stations or _prismatic(), "2D-F", 0.003, **options)


@pytest.fixture(scope="module")
def geometry_blade(laminates):
    return BladeBuilder.from_geometry(_geometry(), n_span=4).build(laminates, FSDTModel())


def test_geometry_mesh_sets_and_normals(geometry_blade):
    blade = geometry_blade
    assert blade.surface == "mid" and blade.geometry is not None
    for name in ("skin", "caps", "webs", "all", "spar_cap_suction_surface", "shear_web_1",
                 "skin_suction_surface", "skin_pressure_surface:leading_edge"):
        assert blade.elements(name), name
    assert set(blade.elements("caps")) <= set(blade.elements("skin"))
    assert set(blade.elements("caps")) <= set(blade.elements("skin_suction_surface")) | set(
        blade.elements("skin_pressure_surface"))
    mesh = blade.mesh
    by_id = {e.element_id: e for e in mesh.shell_elements}
    centre = np.array([0.0, 0.0])
    for eid in blade.elements("skin"):
        e = by_id[eid]
        x = np.mean([mesh.node_coordinates(n)[:2] for n in e.node_ids], axis=0)
        n3 = np.asarray(e.frame)[2]
        assert abs(n3[2]) < 1e-9 and n3[:2] @ (x - centre) > 0.0  # out of the (convex) section
    for eid in blade.elements("webs"):
        assert np.asarray(by_id[eid].frame)[2][0] > 0.99  # towards the trailing edge
    z = [mesh.node_coordinates(n)[2] for n in blade.nodes("tip")]
    assert np.allclose(z, 4.0) and blade.span_positions[-1] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="mid-surface already"):
        blade.to_mid_surface()


def test_geometry_mesh_follows_taper_and_twist(laminates):
    stations = [Station(0.0, 1.2, AIRFOIL, twist=8.0, pitch_axis=0.3),
                Station(4.0, 0.8, AIRFOIL, twist=0.0, pitch_axis=0.3, offset=(0.0, 0.05))]
    geo = _geometry(stations)
    builder = BladeBuilder.from_geometry(geo, n_span=2, element="quad4", wall_elements={"web": 3})
    blade = builder.build(laminates, FSDTModel())
    mesh = blade.mesh
    for z in (0.0, 2.0, 4.0):
        mid = geo.section_at(z).mid_surface()
        nodes = np.array([mesh.node_coordinates(n)[:2] for n in blade.section_nodes(z)])
        # every mesh node lies on the cut's wall mid-surface
        P = mid.points
        A, B = P[[s.start for s in mid.segments]], P[[s.end for s in mid.segments]]
        for p in nodes:
            t = np.clip(np.einsum("ij,ij->i", p - A, B - A) / np.einsum("ij,ij->i", B - A, B - A), 0.0, 1.0)
            assert np.min(np.linalg.norm(A + t[:, None] * (B - A) - p, axis=1)) < 1e-9
    web = [e for e in mesh.shell_elements if e.element_id in set(blade.elements("shear_web_0"))]
    assert len(web) == 3 * 2


def test_geometry_mesh_errors(laminates):
    lib = create_default_library()
    with pytest.raises(ValueError, match="mm thick"):
        BladeBuilder.from_geometry(_geometry(), n_span=1, element="quad4").build(
            dict(laminates, cap=Laminate.from_angles(lib.get("T300/5208"), [0] * 8, 0.5e-3)), FSDTModel())
    stations = [Station(0.0, 2.0, AIRFOIL, pitch_axis=0.3), Station(10.0, 1.0, AIRFOIL, pitch_axis=0.3)]
    dropped = BladeGeometry(stations, "2D-F", 0.004, spar_cap_thickness=0.008, shear_web_thickness=0.004,
                            web_offsets=(0.4, 1.2))
    with pytest.raises(ValueError, match="different walls"):
        BladeBuilder.from_geometry(dropped, n_span=1, element="quad4").build(laminates, FSDTModel())


def test_geometry_mesh_laminates_as_functions_of_span(laminates):
    lib = create_default_library()
    glass = lib.get("E-glass/epoxy")
    geo = BladeGeometry(_prismatic(), "2D-F", lambda z: 0.003 if z < 2.0 else 0.004, spar_cap_thickness=0.008,
                        shear_web_thickness=0.002, web_positions=(0.18, 0.48))
    thin, thick = laminates["skin"], Laminate.from_angles(glass, [0, 45, -45, 0, 0, -45, 45, 0], 0.5e-3)
    blade = BladeBuilder.from_geometry(geo, n_span=[0.0, 0.5, 1.0], element="quad4").build(
        dict(laminates, skin=lambda z: thin if z < 2.0 else thick), FSDTModel())
    t = {round(e.laminate.thickness, 6) for e in blade.mesh.shell_elements
         if e.element_id in set(blade.elements("skin_suction_surface:leading_edge"))}
    assert t == {0.003, 0.004}
