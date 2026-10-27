"""Blade and wing structures: airfoil shell skins, spar caps and shear webs.

``BladeBuilder`` meshes a blade or wing segment between spanwise stations. Each
station has a chord, twist and airfoil (thickness can change along the span). The
skin is split chordwise into named regions (leading-edge panel, spar cap, trailing-
edge panel, ...) on both surfaces, and flat shear webs join the two surfaces at
chord positions that are region boundaries, so webs, caps and skins share nodes.

Axes: z runs along the span, x along the chord (leading edge to trailing edge) and
y through the thickness (suction side positive). The pitch axis sits at x = 0.
Skin normals point out of the section; web normals point to the trailing edge.

Element sets: ``<region>_ss`` (suction side), ``<region>_ps`` (pressure side) and
``<region>`` (both), ``<web>`` for each web, ``skin``, ``webs`` and ``all``.
Node sets: ``root`` and ``tip`` (end sections), ``le``, ``te``, ``<web>`` (web nodes),
``<web>_ss`` / ``<web>_ps`` (web-to-skin joint lines) and ``all``.

Laminates are given per region and web name; ``<region>_ss`` / ``<region>_ps``
override a side. Skins are laid up from the outer mould line by default
(``skin_reference="outer"``): the laminate grows inward from the airfoil surface, as
it does from a blade mould. The skins are then meshed on the laminates' mid-surface
(``mesh_surface="mid"``, the default), so flat facets keep the torsional stiffness
of thick sections; ``mesh_surface="outer"`` keeps the nodes on the mould line and
offsets the laminates instead. A ``Layup`` can be applied afterwards with the element
sets for ply-by-ply definitions (spar-cap ply drops along the span, for example);
call ``blade.to_mid_surface()`` after it so the mid-surface follows the new plies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from blade_geometry_model.blade_geometry import Airfoil, ShearWeb, SkinRegion, Station

from .laminate_kinematics import KinematicModel
from .laminate_mechanics import Laminate


@dataclass
class BladeMesh:
    """The mesh, its named sets and the section geometry used to build it."""

    mesh: "FEAMesh"  # noqa: F821
    node_sets: Dict[str, List[int]] = field(default_factory=dict)
    element_sets: Dict[str, List[int]] = field(default_factory=dict)
    stations: Tuple[Station, ...] = ()
    span_positions: np.ndarray = field(default_factory=lambda: np.zeros(0))  # element-edge sections
    # the same blade on the outer mould line, kept by a mid-surface mesh so that
    # ``to_mid_surface`` can redo the move after a ``Layup`` changes the plies
    mould_line: Optional["BladeMesh"] = field(default=None, repr=False, compare=False)
    _mid_laminates: Dict[int, Laminate] = field(default_factory=dict, repr=False, compare=False)
    _smoothing: Optional[float] = field(default=None, repr=False, compare=False)

    @property
    def surface(self) -> str:
        """``"mid"`` for skins meshed on their mid-surface, ``"outer"`` on the mould line."""
        return "mid" if self.mould_line is not None else "outer"

    def nodes(self, name: str) -> List[int]:
        try:
            return self.node_sets[name]
        except KeyError as exc:
            raise KeyError(f"No node set '{name}'. Available: {', '.join(self.node_sets)}") from exc

    def elements(self, name: str) -> List[int]:
        try:
            return self.element_sets[name]
        except KeyError as exc:
            raise KeyError(f"No element set '{name}'. Available: {', '.join(self.element_sets)}") from exc

    def section_nodes(self, z: float, tol: float = 1e-9) -> List[int]:
        """Nodes of the cross-section at span position ``z`` (a mesh section)."""
        return [n for n in self.node_sets["all"] if abs(self.mesh.node_coordinates(n)[2] - z) <= tol]

    def to_mid_surface(self, smoothing: Optional[float] = None) -> "BladeMesh":
        """The same blade meshed on the skin laminates' mid-surface instead of the mould line.

        Each skin node moves along the section normal by the offset of the laminates
        around it (inward by half the skin thickness for skins on the outer mould line),
        so the skin laminates need no offset. The shift is averaged over the elements at
        each node and then smoothed along the skin with a hat kernel ``smoothing`` long
        (two of the thickest skins by default), so the surface steps gently across
        thickness changes: region boundaries, spar-cap edges and ply drops applied with a
        ``Layup``. Elements in a blend keep the small residual offset that puts their
        plies where they were; elsewhere (and wherever it is under 5% of the
        thickness) the offset is zero. Web nodes are re-spaced
        between their joints, so the webs end on the skins' mid-surfaces.

        Nodes move within the cross-section plane (z is kept), so the root, tip and
        mesh sections stay where they were.

        On a blade that is already on its mid-surface (the ``BladeBuilder`` default),
        this redoes the move from the mould line: elements whose laminate was replaced
        since (by a ``Layup``, referenced to the mould line as laid in the mould) take
        their new plies, the others keep their original ones, and ``smoothing``
        defaults to the length used the first time. So after applying a ``Layup`` to
        a mid-surface blade, call ``blade = blade.to_mid_surface()``.
        """
        return _mid_surface(self, smoothing)


class BladeBuilder:
    """Shell model of a blade or wing segment: skins, spar caps and shear webs.

    Parameters
    ----------
    stations:
        Two or more ``Station`` objects in increasing ``z``. Chord, twist, pitch axis,
        offset and the airfoil surfaces are interpolated linearly between them.
    regions:
        ``SkinRegion`` objects that tile the chord from 0 (LE) to 1 (TE) without gaps;
        the same regions are used on both surfaces.
    webs:
        ``ShearWeb`` objects at region boundaries (chord fractions strictly inside).
    n_span:
        Elements along the span (an int), or increasing span fractions 0..1.
    element:
        ``"quad8_curved"`` (default), ``"quad8"`` or ``"quad4"``. Quadratic elements
        put their mid-side nodes on the airfoil surface. Flat facets (``"quad4"``,
        ``"quad8"``) under-predict the torsional stiffness of thick-walled sections
        if the skins stay on the mould line with offset laminates
        (``mesh_surface="outer"``); on the mid-surface (the default) they do not.
    trailing_edge:
        ``"auto"`` joins the surfaces at a sharp trailing edge and adds a flat
        ``te_web`` across a blunt one; ``"web"`` always adds it.
    material_axis:
        The 0-degree ply direction, projected onto each element: the span axis by
        default, so 0-degree plies run along the blade. Ply angles turn about each
        element's normal (outward on the skins), so the same angle on the suction and
        pressure sides is mirrored when seen from one side of the blade: equal angles
        give extension-twist coupling, opposite angles bend-twist coupling.
    mesh_surface:
        ``"mid"`` (default) meshes the skins on the laminates' mid-surface, smoothed
        across thickness changes over ``mid_surface_smoothing`` (see
        ``BladeMesh.to_mid_surface``), with the webs between the skins'
        mid-surfaces. Every element type then keeps the torsional stiffness of thick
        closed sections. ``"outer"`` meshes the skins on the outer mould line and
        offsets their laminates (see ``skin_reference``), which keeps the true outer
        geometry but makes flat facets (``"quad4"``, ``"quad8"``) too soft in torsion
        where the skins are thick. After a ``Layup``, call ``blade.to_mid_surface()``
        so the mid-surface follows the new plies.
    """

    def __init__(
        self,
        stations: Sequence[Station],
        regions: Sequence[SkinRegion],
        webs: Sequence[ShearWeb] = (),
        n_span: Union[int, Sequence[float]] = 10,
        element: str = "quad8_curved",
        skin_reference: str = "outer",
        trailing_edge: str = "auto",
        te_web_elements: int = 2,
        material_axis=(0.0, 0.0, 1.0),
        mesh_surface: str = "mid",
        mid_surface_smoothing: Optional[float] = None,
    ):
        self.material_axis = material_axis
        if mesh_surface not in ("outer", "mid"):
            raise ValueError("mesh_surface must be 'outer' or 'mid'")
        self.mesh_surface = mesh_surface
        self.mid_surface_smoothing = mid_surface_smoothing
        self.stations = tuple(sorted(stations, key=lambda s: s.z))
        if len(self.stations) < 2:
            raise ValueError("a blade needs at least two stations")
        if np.any(np.diff([s.z for s in self.stations]) <= 0.0):
            raise ValueError("stations need distinct span positions")
        self.regions = tuple(sorted(regions, key=lambda r: r.start))
        if not self.regions or abs(self.regions[0].start) > 1e-12 or abs(self.regions[-1].end - 1.0) > 1e-12:
            raise ValueError("skin regions must cover the chord from 0 to 1")
        for a, b in zip(self.regions[:-1], self.regions[1:]):
            if abs(a.end - b.start) > 1e-12:
                raise ValueError(f"regions '{a.name}' and '{b.name}' leave a gap or overlap")
        names = [r.name for r in self.regions]
        if len(set(names)) != len(names):
            raise ValueError("region names must be unique")
        bounds = {r.start for r in self.regions[1:]}
        self.webs = tuple(sorted(webs, key=lambda w: w.position))
        for w in self.webs:
            if not any(abs(w.position - b) < 1e-12 for b in bounds):
                raise ValueError(f"web '{w.name}' at {w.position} must sit on a region boundary {sorted(bounds)}")
        if element not in ("quad4", "quad8", "quad8_curved"):
            raise ValueError("element must be 'quad4', 'quad8' or 'quad8_curved'")
        if skin_reference not in ("outer", "mid", "inner"):
            raise ValueError("skin_reference must be 'outer', 'mid' or 'inner'")
        if trailing_edge not in ("auto", "web"):
            raise ValueError("trailing_edge must be 'auto' or 'web'")
        self.element = element
        self.skin_reference = skin_reference
        self.trailing_edge = trailing_edge
        self.te_web_elements = int(te_web_elements)
        if np.ndim(n_span) == 0:
            self.span_fractions = np.linspace(0.0, 1.0, int(n_span) + 1)
        else:
            self.span_fractions = np.asarray(n_span, dtype=float)
        f = self.span_fractions
        if f.size < 2 or abs(f[0]) > 1e-12 or abs(f[-1] - 1.0) > 1e-12 or np.any(np.diff(f) <= 0):
            raise ValueError("n_span must be >= 1 or increasing fractions from 0 to 1")

    # ------------------------------------------------------------ geometry
    def _station_at(self, z: float):
        zs = np.array([s.z for s in self.stations])
        k = int(np.clip(np.searchsorted(zs, z, side="right") - 1, 0, len(zs) - 2))
        a, b = self.stations[k], self.stations[k + 1]
        w = float(np.clip((z - a.z) / (b.z - a.z), 0.0, 1.0))
        lerp = lambda p, q: (1 - w) * p + w * q  # noqa: E731
        return a, b, w, lerp(a.chord, b.chord), lerp(a.twist, b.twist), lerp(a.pitch_axis, b.pitch_axis), \
            lerp(np.asarray(a.offset, float), np.asarray(b.offset, float))

    def point(self, side: str, x_c: float, z: float) -> np.ndarray:
        """Global position of the chord fraction ``x_c`` on ``side`` ("upper"/"lower") at span z."""
        a, b, w, chord, twist, pitch, offset = self._station_at(z)
        y_c = (1 - w) * a.airfoil.surface(side, x_c) + w * b.airfoil.surface(side, x_c)
        return self._place(x_c, float(y_c), chord, twist, pitch, offset, z)

    @staticmethod
    def _place(x_c, y_c, chord, twist, pitch, offset, z) -> np.ndarray:
        x, y = (x_c - pitch) * chord, y_c * chord
        t = np.radians(twist)
        c, s = np.cos(t), np.sin(t)  # positive twist is nose up: the LE (x < 0) moves to +y
        return np.array([c * x + s * y + offset[0], -s * x + c * y + offset[1], z])

    def te_gap(self, z: float) -> float:
        return float(np.linalg.norm(self.point("upper", 1.0, z) - self.point("lower", 1.0, z)))

    # ---------------------------------------------------------------- build
    def _chord_parameters(self, quadratic: bool):
        """Per chordwise node index: (region, s in [0, 1]); and each region's index span."""
        out = [(self.regions[0], 0.0)]
        spans = {}
        for r in self.regions:
            n = r.n * (2 if quadratic else 1)
            start = len(out) - 1
            out.extend((r, float(v)) for v in np.linspace(0.0, 1.0, n + 1)[1:])
            spans[r.name] = (start, start + n)
        return out, spans

    def _x_at(self, side: str, region: SkinRegion, s: float, z: float) -> float:
        """Chord fraction at arc-length fraction ``s`` of ``region`` on ``side`` (equal element lengths)."""
        if s <= 0.0:
            return region.start
        if s >= 1.0:
            return region.end
        key = (side, region.name, round(z, 12))
        table = self._arc_cache.get(key)
        if table is None:
            # sample densely, finer towards the LE where the surface turns fastest
            u = np.linspace(0.0, 1.0, 801)
            x = region.start + (region.end - region.start) * (1.0 - np.cos(0.5 * np.pi * u) if region.start == 0.0 else u)
            pts = np.array([self.point(side, float(v), z) for v in x])
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            table = (arc / arc[-1], x)
            self._arc_cache[key] = table
        return float(np.interp(s, *table))

    def build(self, laminates, kinematics: KinematicModel, **element_options) -> BladeMesh:
        """Mesh the blade; ``laminates`` maps region / web names (or ``"<region>_ss"``) to laminates.

        One ``Laminate`` for everything is accepted too.
        """
        from .laminate_solver_pipeline import FEAMesh

        element_options = dict(element_options)
        element_options.setdefault("material_axis", self.material_axis)
        quadratic = self.element != "quad4"
        step = 2 if quadratic else 1
        etype = "quad8_curved" if self.element == "quad8_curved" else None
        self._arc_cache = {}
        params, spans = self._chord_parameters(quadratic)
        xc = np.array([r.start + v * (r.end - r.start) for r, v in params])  # nominal, for web lookup
        sf = self.span_fractions
        z0, z1 = self.stations[0].z, self.stations[-1].z
        zf = sf if not quadratic else np.concatenate([[sf[0]], np.column_stack([0.5 * (sf[:-1] + sf[1:]), sf[1:]]).ravel()])
        z_nodes = z0 + zf * (z1 - z0)
        nz = len(sf) - 1
        mesh = FEAMesh()
        node = 1
        grid: Dict[Tuple[str, int, int], int] = {}
        sharp = self.trailing_edge == "auto" and max(self.te_gap(z0), self.te_gap(z1)) < 1e-9 * max(
            s.chord for s in self.stations)
        imax = len(xc) - 1
        for j, z in enumerate(z_nodes):
            for i, (region, sv) in enumerate(params):
                if quadratic and i % 2 and j % 2:
                    continue
                for side in ("upper", "lower"):
                    if side == "lower" and (i == 0 or (sharp and i == imax)):
                        grid[(side, i, j)] = grid[("upper", i, j)]
                        continue
                    mesh.add_node(node, self.point(side, self._x_at(side, region, sv, float(z)), float(z)))
                    grid[(side, i, j)] = node
                    node += 1

        elements: Dict[str, List[int]] = {}
        eid = 1

        def lam_for(name: str, side: Optional[str], skin: bool) -> Laminate:
            if isinstance(laminates, Laminate):
                lam = laminates
            else:
                key = f"{name}_{side}" if side else name
                lam = laminates.get(key, laminates.get(name)) if hasattr(laminates, "get") else None
                if lam is None:
                    raise KeyError(f"No laminate for '{key}'. Give one for '{name}' or '{key}'")
            if skin and self.skin_reference != "mid":
                h = lam.thickness
                lam = Laminate(lam.plies, lam.name, offset=-0.5 * h if self.skin_reference == "outer" else 0.5 * h)
            return lam

        def quad(nodes_ij, side):
            """Corner + mid-side node ids of the cell with corner grid indices (i, j)."""
            i, j = nodes_ij
            g = lambda a, b: grid[(side, a, b)]  # noqa: E731
            d = step
            corners = [g(i, j), g(i + d, j), g(i + d, j + d), g(i, j + d)]
            if not quadratic:
                return corners
            return corners + [g(i + 1, j), g(i + d, j + 1), g(i + 1, j + d), g(i, j + 1)]

        def flip(conn):
            if len(conn) == 4:
                return [conn[0], conn[3], conn[2], conn[1]]
            return [conn[0], conn[3], conn[2], conn[1], conn[7], conn[6], conn[5], conn[4]]

        for r in self.regions:
            a, b = spans[r.name]
            for side, tag in (("upper", "ss"), ("lower", "ps")):
                lam = lam_for(r.name, tag, True)
                for jj in range(nz):
                    for ii in range(a, b, step):
                        conn = quad((ii, step * jj), side)
                        # chordwise x span: (+chord) x (+z) points into the section on the upper side
                        if side == "upper":
                            conn = flip(conn)
                        mesh.add_element(eid, conn, lam, kinematics, element_type=etype, **element_options)
                        elements.setdefault(f"{r.name}_{tag}", []).append(eid)
                        elements.setdefault(r.name, []).append(eid)
                        elements.setdefault("skin", []).append(eid)
                        eid += 1

        web_nodes: Dict[str, List[int]] = {}
        joint: Dict[str, List[int]] = {}
        web_defs = list(self.webs)
        if not sharp:
            web_defs.append(ShearWeb("te_web", 1.0, self.te_web_elements))
        for web in web_defs:
            i = int(np.argmin(np.abs(xc - web.position)))
            lam = lam_for(web.name, None, False)
            m = web.n * step
            col: Dict[Tuple[int, int], int] = {}
            for j, z in enumerate(z_nodes):
                top, bot = mesh.node_coordinates(grid[("upper", i, j)]), mesh.node_coordinates(grid[("lower", i, j)])
                for k in range(m + 1):
                    if quadratic and k % 2 and j % 2:
                        continue
                    if k == 0:
                        col[(k, j)] = grid[("lower", i, j)]
                    elif k == m:
                        col[(k, j)] = grid[("upper", i, j)]
                    else:
                        mesh.add_node(node, bot + (top - bot) * k / m)
                        col[(k, j)] = node
                        node += 1
            for jj in range(nz):
                for kk in range(0, m, step):
                    j = step * jj
                    g = lambda a, b: col[(a, b)]  # noqa: E731
                    d = step
                    conn = [g(kk, j), g(kk + d, j), g(kk + d, j + d), g(kk, j + d)]
                    if quadratic:
                        conn += [g(kk + 1, j), g(kk + d, j + 1), g(kk + 1, j + d), g(kk, j + 1)]
                    # lower -> upper (+y) x (+z) points to +x (the TE) for an untwisted section
                    mesh.add_element(eid, conn, lam, kinematics, element_type=etype, **element_options)
                    elements.setdefault(web.name, []).append(eid)
                    elements.setdefault("webs", []).append(eid)
                    eid += 1
            web_nodes[web.name] = list(dict.fromkeys(col.values()))
            joint[f"{web.name}_ss"] = [col[(m, j)] for j in range(len(z_nodes))]
            joint[f"{web.name}_ps"] = [col[(0, j)] for j in range(len(z_nodes))]

        all_nodes = mesh.node_ids
        sets: Dict[str, List[int]] = {
            "root": [n for n in all_nodes if abs(mesh.node_coordinates(n)[2] - z0) < 1e-12 * max(1.0, abs(z1))],
            "tip": [n for n in all_nodes if abs(mesh.node_coordinates(n)[2] - z1) < 1e-12 * max(1.0, abs(z1))],
            "le": list(dict.fromkeys(grid[("upper", 0, j)] for j in range(len(z_nodes)))),
            "te": list(dict.fromkeys(v for j in range(len(z_nodes)) for v in
                                     (grid[("upper", imax, j)], grid[("lower", imax, j)]))),
            "all": all_nodes,
        }
        sets.update(web_nodes)
        sets.update(joint)
        elements["all"] = mesh.element_ids
        blade = BladeMesh(mesh, sets, elements, self.stations, z0 + sf * (z1 - z0))
        if self.mesh_surface == "mid":
            blade = blade.to_mid_surface(self.mid_surface_smoothing)
        return blade


# ---------------------------------------------------------------------------
# Mid-surface meshes
# ---------------------------------------------------------------------------
def _node_normals(mesh, element_ids) -> Dict[int, np.ndarray]:
    """Area-weighted mean element normal at each node, projected into the section plane."""
    acc: Dict[int, np.ndarray] = {}
    for eid in element_ids:
        e = mesh.element(eid)
        n = np.asarray(e.frame)[2] * e.area
        for nid in e.node_ids:
            acc[nid] = acc.get(nid, 0.0) + n
    out = {}
    for nid, v in acc.items():
        v = np.array([v[0], v[1], 0.0])
        out[nid] = v / max(np.linalg.norm(v), 1e-300)
    return out


def _smooth_on_surface(mesh, element_ids, values: Dict[int, float], weights: Dict[int, float],
                       length: float) -> Dict[int, float]:
    """Hat-kernel average of nodal values over surface paths shorter than ``length``."""
    if length <= 0.0:
        return dict(values)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
    except ImportError:  # pragma: no cover - SciPy is an optional dependency
        return dict(values)
    ids = sorted(values)
    index = {n: k for k, n in enumerate(ids)}
    rows, cols, dist = [], [], []
    for eid in element_ids:
        nodes = [n for n in mesh.element(eid).node_ids if n in index]
        for a in nodes:
            for b in nodes:
                if a < b:
                    d = float(np.linalg.norm(mesh.node_coordinates(a) - mesh.node_coordinates(b)))
                    rows += [index[a], index[b]]
                    cols += [index[b], index[a]]
                    dist += [d, d]
    graph = coo_matrix((dist, (rows, cols)), shape=(len(ids), len(ids))).tocsr()
    w = np.array([weights[n] for n in ids])
    v = np.array([values[n] for n in ids])
    out = np.empty(len(ids))
    for start in range(0, len(ids), 256):  # in blocks: the distance matrix is dense
        block = np.arange(start, min(start + 256, len(ids)))
        D = dijkstra(graph, directed=False, indices=block, limit=length)
        W = np.where(np.isfinite(D), 1.0 - D / length, 0.0) * w[None, :]
        out[block] = (W @ v) / W.sum(axis=1)
    return dict(zip(ids, out))


RESIDUAL_OFFSET_TOL = 0.05  # residual offsets below this fraction of the thickness are dropped


def _mid_surface(blade: BladeMesh, smoothing: Optional[float]) -> BladeMesh:
    if blade.mould_line is None:
        return _move_to_mid_surface(blade, smoothing)
    # redo the move from the mould line, with the plies of any laminate replaced since
    source = blade.mould_line
    if smoothing is None:
        smoothing = blade._smoothing
    outer = _copy_mesh(source.mesh)
    for e in blade.mesh.elements:
        if blade._mid_laminates.get(e.element_id) is not e.laminate:
            outer.replace_element(outer.element(e.element_id).with_laminate(e.laminate))
    return _move_to_mid_surface(BladeMesh(outer, {k: list(v) for k, v in source.node_sets.items()},
                                          {k: list(v) for k, v in source.element_sets.items()},
                                          source.stations, source.span_positions), smoothing)


def _copy_mesh(mesh):
    from .laminate_solver_pipeline import FEAMesh

    new = FEAMesh(mesh.hygrothermal_engine, mesh.element_registry, mesh.solid_registry)
    for n in mesh.node_ids:
        new.add_node(n, mesh.node_coordinates(n).copy())
    for e in mesh.elements:
        new.insert_element(e)
    return new


def _move_to_mid_surface(blade: BladeMesh, smoothing: Optional[float]) -> BladeMesh:
    from .laminate_solver_pipeline import FEAMesh

    mesh = blade.mesh
    skin = list(blade.element_sets.get("skin", []))
    if not skin:
        return blade
    thick = max(mesh.element(e).laminate.thickness for e in skin)
    length = 2.0 * thick if smoothing is None else float(smoothing)

    # the shift of each skin node: the offset of the laminates around it, area-weighted
    normals = _node_normals(mesh, skin)
    total: Dict[int, float] = {}
    area: Dict[int, float] = {}
    for eid in skin:
        e = mesh.element(eid)
        w = e.area / len(e.node_ids)
        for nid in e.node_ids:
            total[nid] = total.get(nid, 0.0) + w * e.laminate.offset
            area[nid] = area.get(nid, 0.0) + w
    shift = {n: total[n] / area[n] for n in total}
    shift = _smooth_on_surface(mesh, skin, shift, area, length)

    xyz = {n: mesh.node_coordinates(n).copy() for n in mesh.node_ids}
    for n, d in shift.items():
        xyz[n] = xyz[n] + d * normals[n]

    # web nodes: same fraction of the way between their (moved) joints at the same z
    skin_nodes = set(shift)
    for name in blade.node_sets:
        if f"{name}_ss" not in blade.node_sets or f"{name}_ps" not in blade.node_sets:
            continue
        top = {round(float(mesh.node_coordinates(n)[2]), 9): n for n in blade.node_sets[f"{name}_ss"]}
        bot = {round(float(mesh.node_coordinates(n)[2]), 9): n for n in blade.node_sets[f"{name}_ps"]}
        for n in blade.node_sets[name]:
            if n in skin_nodes:
                continue
            p = mesh.node_coordinates(n)
            z = round(float(p[2]), 9)
            a, b = mesh.node_coordinates(bot[z]), mesh.node_coordinates(top[z])
            f = float(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-300))
            xyz[n] = xyz[bot[z]] + f * (xyz[top[z]] - xyz[bot[z]])

    new = FEAMesh(mesh.hygrothermal_engine, mesh.element_registry, mesh.solid_registry)
    for n in mesh.node_ids:
        new.add_node(n, xyz[n])
    skin_set = set(skin)
    for e in mesh.elements:
        lam = e.laminate
        if e.element_id in skin_set:
            # the part of the node moves along this element's normal
            e3 = np.asarray(e.frame)[2]
            moved = float(np.mean([shift[n] * float(np.dot(normals[n], e3)) for n in e.node_ids]))
            residual = lam.offset - moved
            # residuals under 5% of the thickness are rounding of the smoothing: dropping
            # them keeps neighbouring elements on one shell section (CalculiX expands
            # shells with differing offsets at a node into knots, and runs out of nodes
            # when there are many)
            lam = lam.with_offset(0.0 if abs(residual) < RESIDUAL_OFFSET_TOL * lam.thickness else residual)
        coords = np.array([xyz[n] for n in e.node_ids])
        try:
            moved_element = type(e)(e.element_id, e.node_ids, coords, lam, e._kinematics, **e._options())
        except ValueError as exc:
            raise ValueError(f"Element {e.element_id} folds over on the mid-surface (a narrow element between "
                             f"skins of different thickness); smooth over a longer length than {length:g} "
                             "(mid_surface_smoothing) or widen the element") from exc
        new.insert_element(moved_element)
    return BladeMesh(new, {k: list(v) for k, v in blade.node_sets.items()},
                     {k: list(v) for k, v in blade.element_sets.items()}, blade.stations, blade.span_positions,
                     mould_line=BladeMesh(_copy_mesh(mesh), blade.node_sets, blade.element_sets, blade.stations,
                                          blade.span_positions),
                     _mid_laminates={e.element_id: e.laminate for e in new.elements}, _smoothing=smoothing)


# ---------------------------------------------------------------------------
# Beam section properties
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BeamSectionProperties:
    """Beam properties of a blade segment from the shell model, in global axes.

    ``stiffness_6x6`` is the Timoshenko section stiffness: it relates the section forces
    [Fx, Fy, N, Mx, My, Mz] to the beam strains [gamma_x, gamma_y, eps, kappa_x,
    kappa_y, kappa_z] about the reference point (the pitch axis by default), in the
    BeamDyn / windIO order, with gamma_x = u_x' - theta_y and gamma_y = u_y' + theta_x.
    Its off-diagonal terms are the couplings: extension-bending, bend-twist (for example
    of an aeroelastically tailored blade), shear-twist (the shear centre away from the
    reference) and shear-bending.

    ``stiffness`` is the classical (Euler-Bernoulli) 4 x 4 for [N, Mx, My, Mz] against
    [eps, kappa_x, kappa_y, kappa_z]: the 6 x 6 with the shear forces free (the inverse
    of the axial-bending-torsion block of the 6 x 6 compliance), as an Euler-Bernoulli
    beam code wants it.

    ``stiffness_7x7`` (when measured, ``warping=True``) adds the warping DOF of
    Timoshenko-Vlasov beam theory: the seventh strain is kappa_z' and its force the
    bimoment, with ``EGamma`` on the diagonal; ``warping_shear_stiffness`` is the
    extra shear stiffness S of Benscoter's theory (closed sections), so that a
    ``RestrainedTorsionBeam(GJ, EGamma, L, S)`` reproduces the shell's restrained twist.

    ``tension_centre`` is where an axial force causes no bending and ``shear_centre``
    where a shear force causes no twist. Lengths are in the mesh units.
    """

    z_range: Tuple[float, float]
    reference: Tuple[float, float]
    stiffness: np.ndarray  # 4 x 4 classical
    tension_centre: Tuple[float, float]
    shear_centre: Tuple[float, float]
    mass_per_length: float
    mass_centre: Tuple[float, float]
    stiffness_6x6: Optional[np.ndarray] = None
    stiffness_7x7: Optional[np.ndarray] = None
    warping_shear_stiffness: Optional[float] = None

    @property
    def EA(self) -> float:
        return float(self.stiffness[0, 0])

    @property
    def EI_x(self) -> float:
        """Bending stiffness about the x axis (curvature in the y-z plane)."""
        return float(self.stiffness[1, 1])

    @property
    def EI_y(self) -> float:
        return float(self.stiffness[2, 2])

    @property
    def GJ(self) -> float:
        return float(self.stiffness[3, 3])

    @property
    def GA(self) -> Tuple[float, float]:
        """Transverse shear stiffnesses (GA_x, GA_y): the 6 x 6 diagonal for gamma_x and gamma_y."""
        return float(self.stiffness_6x6[0, 0]), float(self.stiffness_6x6[1, 1])

    @property
    def EGamma(self) -> Optional[float]:
        """Warping stiffness EΓ (with ``warping=True``)."""
        return None if self.stiffness_7x7 is None else float(self.stiffness_7x7[6, 6])

    @property
    def compliance_6x6(self) -> np.ndarray:
        return np.linalg.inv(self.stiffness_6x6)

    @property
    def bend_twist_coupling(self) -> Tuple[float, float]:
        """Normalised couplings S[1,3]/sqrt(S11 S33) and S[2,3]/sqrt(S22 S33) (zero when uncoupled)."""
        S = self.stiffness
        return (float(S[1, 3] / np.sqrt(S[1, 1] * S[3, 3])), float(S[2, 3] / np.sqrt(S[2, 2] * S[3, 3])))

    def at(self, point: Tuple[float, float]) -> np.ndarray:
        """The 6 x 6 moved to another reference point (same axes)."""
        return transform_beam_stiffness(self.stiffness_6x6, (point[0] - self.reference[0], point[1] - self.reference[1]))


def transform_beam_stiffness(K6: np.ndarray, offset: Tuple[float, float], angle: float = 0.0) -> np.ndarray:
    """Move a 6 x 6 Timoshenko stiffness (BeamDyn order) to a reference point ``offset``
    (x, y) from the current one, and turn its axes by ``angle`` (radians, about z).

    Forces at the new point: F' = F, M' = M - r x F, written f' = T f; the strains
    transform so that the work f . e is unchanged (e = T^T e'), so K' = T K T^T.
    """
    dx, dy = float(offset[0]), float(offset[1])
    T = np.eye(6)
    # M' = M - r x F with r = (dx, dy, 0): (r x F) = (dy Fz, -dx Fz, dx Fy - dy Fx)
    T[3, 2] -= dy
    T[4, 2] += dx
    T[5, 0] += dy
    T[5, 1] -= dx
    if angle:
        c, s = np.cos(angle), np.sin(angle)
        R2 = np.array([[c, s], [-s, c]])  # components in axes turned by +angle
        Rot = np.eye(6)
        Rot[0:2, 0:2] = R2
        Rot[3:5, 3:5] = R2
        T = Rot @ T
    return T @ K6 @ T.T


def beam_section_properties(
    blade: BladeMesh,
    z_range: Optional[Tuple[float, float]] = None,
    reference: Optional[Tuple[float, float]] = None,
    solver=None,
    warping: bool = False,
    section_weighting: str = "stiffness",
    end_ribs: float = 0.0,
) -> BeamSectionProperties:
    """Extract beam properties of a blade segment by loading the shell model as a cantilever.

    The root is clamped and unit loads are applied through a rigid tip section (RBE2
    on the translations, like an end rib). Two more distributing links measure the mean
    section motion (translations and rotations) at the ends of ``z_range`` (the middle
    fifth of the span by default), away from the end effects, and their differences give
    the strains of the beam. Axial force and moments are constant along a cantilever, so
    they give four columns of the 6 x 6 compliance directly; the tip shear forces add a
    bending moment that varies along the span, which is subtracted to give the shear
    columns (the shear strain is the slope of the axis minus the section rotation,
    averaged over ``z_range`` exactly). The blade should be prismatic over ``z_range``
    (a long, uniform segment, five chords or more, gives the cleanest values).
    ``reference`` is the (x, y) point the properties refer to.

    The section rotation of a Timoshenko beam is a definition: shear warps the section,
    so it has no single rotation. ``section_weighting="stiffness"`` (the default) takes
    the axial-stiffness-weighted least-squares rotation (each node weighted by Et times
    its share of the section contour), the rotation that does work with the bending
    moment, as 2D cross-section tools (VABS, ANBA, BECAS) define it. ``"uniform"``
    weights every node alike. The 4 x 4 is the same either way (pure bending keeps the
    section plane); the shear stiffnesses are not.

    The clamped root and the rigid tip restrain warping. Left free to distort, the
    section relieves that restraint by distorting, over a length that is long for a
    section with no ribs, and the distortion reaches the middle of the span and shifts
    GJ by several percent (examples 27 and 28). ``end_ribs`` (a fraction of the span)
    holds the sections within that distance of either end rigid in their plane, as ribs
    or bulkheads would, so that only the short warping boundary layer remains and the
    middle of the span carries the Saint-Venant state: GJ then matches
    ``ThinWalledSection.from_blade_mesh(blade).GJ`` (``anisotropic_section_model``). The ribs leave the sections free to
    warp and the middle of the span free to distort. ``end_ribs=0`` (the default) gives
    the bare cantilever. On a faceted curved section, flat Quad8 elements (not Quad4 or
    curved Quad8) deform in their plane under torsion and the ribs stiffen them (a thick
    tube reads 5% high), so use ribs with Quad4 or curved Quad8 there.

    ``warping=True`` also measures the warping stiffness for ``stiffness_7x7``: one more
    run holds every mesh section rigid in its plane (free to warp, as Vlasov and
    Benscoter assume), clamps the root and twists the tip, and Benscoter's closed form
    is fitted to the twist along the span with GJ from the 6 x 6, giving EΓ and the
    warping shear stiffness S.
    """
    from .laminate_connectors import add_reference_node, distributing_link, rigid_link
    from .laminate_solver_pipeline import BoundaryConditions, LoadCase, StructuralGlobalSolver

    mesh = blade.mesh.copy()
    zs = np.asarray(blade.span_positions)
    z0, z1 = float(zs[0]), float(zs[-1])
    if z_range is None:
        z_range = (z0 + 0.4 * (z1 - z0), z0 + 0.6 * (z1 - z0))
    za, zb = (float(zs[int(np.argmin(np.abs(zs - z)))]) for z in z_range)
    if zb <= za:
        raise ValueError("z_range must span at least one mesh section")
    ref = (0.0, 0.0) if reference is None else (float(reference[0]), float(reference[1]))
    bcs = BoundaryConditions().clamp(blade.nodes("root"))
    tip = add_reference_node(mesh, (ref[0], ref[1], z1))
    rigid_link(mesh, bcs, tip, blade.nodes("tip"), dofs=(0, 1, 2))  # a rigid tip diaphragm
    if end_ribs > 0.0:
        for z in zs[1:-1]:
            if z - z0 <= end_ribs * (z1 - z0) or z1 - z <= end_ribs * (z1 - z0):
                rib = add_reference_node(mesh, (ref[0], ref[1], float(z)))
                rigid_link(mesh, bcs, rib, blade.section_nodes(float(z)), dofs=(0, 1))
                bcs.fix([rib], [2, 3, 4])
    probes = []
    for z in (za, zb):
        n = add_reference_node(mesh, (ref[0], ref[1], z))
        nodes = blade.section_nodes(z)
        if section_weighting == "stiffness":
            weights = _section_weights(mesh, z, nodes)
        elif section_weighting == "uniform":
            weights = None
        else:
            raise ValueError("section_weighting must be 'stiffness' or 'uniform'")
        distributing_link(mesh, bcs, n, nodes, weights)
        probes.append(n)
    cases = []
    for dof in (2, 3, 4, 5, 0, 1):  # Fz, Mx, My, Mz, then shear Fx, Fy
        cases.append(LoadCase(f"unit_{dof}").add_nodal_load(tip, dof, 1.0))
    solver = solver or StructuralGlobalSolver()
    sols = solver.solve_many(mesh, bcs, cases)
    L = zb - za

    def strains(sol, curvature_gradient=(0.0, 0.0)):
        """Mean [gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z] over z_range.

        The mean section rotation is the trapezoidal mean of the end rotations, exact
        for a linear rotation and corrected by -L^2/12 theta'' when the curvature varies
        linearly (``curvature_gradient`` = (kappa_x', kappa_y')).
        """
        ua, ub = sol.node_displacement(probes[0]), sol.node_displacement(probes[1])
        d = (ub - ua) / L
        th = 0.5 * (ua[3:5] + ub[3:5]) - L**2 / 12.0 * np.asarray(curvature_gradient)
        return np.array([d[0] - th[1], d[1] + th[0], d[2], d[3], d[4], d[5]])

    C = np.zeros((6, 6))  # compliance, BeamDyn order
    for k, sol in zip((2, 3, 4, 5), sols[:4]):
        C[:, k] = strains(sol)
    arm = z1 - 0.5 * (za + zb)
    for k, sol, dM in ((0, sols[4], np.array([0.0, -1.0])), (1, sols[5], np.array([1.0, 0.0]))):
        # unit Fx: My = (z1 - z), dMy/dz = -1; unit Fy: Mx = -(z1 - z), dMx/dz = +1
        grad = C[3:5, 3:5] @ dM
        e = strains(sol, grad)
        M_mid = np.array([-arm, 0.0]) if k == 1 else np.array([0.0, arm])  # [Mx, My] at mid z_range
        C[:, k] = e - C[:, 3:5] @ M_mid
    C = 0.5 * (C + C.T)
    K6 = np.linalg.inv(C)
    K6 = 0.5 * (K6 + K6.T)
    C4 = C[2:, 2:]
    S = np.linalg.inv(C4)
    S = 0.5 * (S + S.T)
    # tension centre: Fz at (x, y) adds Mx = y Fz, My = -x Fz; no curvature there
    Mt = np.column_stack([-C4[1:3, 2], C4[1:3, 1]])
    xt, yt = np.linalg.solve(Mt, -C4[1:3, 0])
    # shear centre: a shear force there causes no twist (Fx at (x, y) adds Mz = -y Fx, Fy adds x Fy)
    x_sc = -C[5, 1] / C[5, 5]
    y_sc = C[5, 0] / C[5, 5]
    # mass per unit length between the probes
    mass, moment = 0.0, np.zeros(2)
    for e in mesh.shell_elements:
        c = np.asarray(e.centroid)
        if za - 1e-12 <= c[2] <= zb + 1e-12:
            rho_a = sum(p.thickness * p.material.density for p in e.laminate.plies)
            mid = c + e.laminate.mid_plane * np.asarray(e.frame)[2]
            mass += rho_a * e.area
            moment += rho_a * e.area * mid[:2]
    m = mass / L
    centre = tuple(moment / mass) if mass > 0 else (float("nan"), float("nan"))
    K7, S_w = None, None
    if warping:
        EGamma, S_w, _ = _fit_warping_stiffness(blade, (x_sc + ref[0], y_sc + ref[1]), solver)
        K7 = np.zeros((7, 7))
        K7[:6, :6] = K6
        K7[6, 6] = EGamma
    return BeamSectionProperties((za, zb), ref, S, (float(xt + ref[0]), float(yt + ref[1])),
                                 (float(x_sc + ref[0]), float(y_sc + ref[1])), float(m), centre, K6, K7, S_w)


def _section_weights(mesh, z: float, nodes: Sequence[int]) -> List[float]:
    """Axial stiffness share of each section node: Et (= 1 / (A^-1)_11 along the beam axis)
    times its share of the element edges on the section (Simpson weights on quadratic edges)."""
    from .laminate_kinematics import integrate_abd

    weight = {int(n): 0.0 for n in nodes}
    scale = max(1.0, abs(z))
    cache: Dict[int, float] = {}
    for e in mesh.shell_elements:
        ids = list(e.node_ids)
        X = np.array([mesh.node_coordinates(n) for n in ids])
        on = np.abs(X[:, 2] - z) <= 1e-9 * scale
        if on.sum() < 2:
            continue
        key = id(e.laminate)
        if key not in cache:
            a = np.linalg.inv(integrate_abd(Laminate(e.laminate.plies))[:3, :3])  # membrane, wall not bending
            frame = np.asarray(e.frame)
            c2 = frame[0, 2] ** 2  # material axis 1 along the beam axis: c2 = 1
            cache[key] = 1.0 / (c2 * a[0, 0] + (1.0 - c2) * a[1, 1])
        Et = cache[key]
        nc = 4 if len(ids) in (4, 8) else 3
        for k in range(nc):
            a_, b_ = k, (k + 1) % nc
            if not (on[a_] and on[b_]):
                continue
            length = float(np.linalg.norm(X[b_] - X[a_]))
            if len(ids) == 2 * nc:
                for n, f in ((ids[a_], 1.0 / 6.0), (ids[nc + k], 4.0 / 6.0), (ids[b_], 1.0 / 6.0)):
                    if n in weight:
                        weight[n] += Et * length * f
            else:
                for n in (ids[a_], ids[b_]):
                    if n in weight:
                        weight[n] += 0.5 * Et * length
    return [weight[int(n)] for n in nodes]


def _fit_warping_stiffness(blade: BladeMesh, shear_centre, solver) -> Tuple[float, Optional[float], float]:
    """EΓ and Benscoter's S from the restrained twist of the shell with rigid sections.

    Every mesh section is tied rigid in its plane (free to warp) to a node on the shear
    centre, the root is clamped and a unit torque twists the tip. Benscoter's closed form
    for a cantilever, GJ theta(z) / T = (1 - rho) z + rho (z - (sinh k(z - L) + sinh kL)
    / (k cosh kL)) with rho = S / (GJ + S), k^2 = GJ S / (EΓ (GJ + S)), is fitted to the
    twist (a scan on k, with GJ and rho by least squares); rho near 1 means Vlasov's
    theory holds (no shear strain from warping; S is then returned as ``None``). Returns
    (EΓ, S, GJ of the rigid-section run).
    """
    from .laminate_connectors import add_reference_node, rigid_link
    from .laminate_solver_pipeline import BoundaryConditions, LoadCase

    mesh = blade.mesh.copy()
    zs = np.asarray(blade.span_positions)
    z0 = float(zs[0])
    bcs = BoundaryConditions().clamp(blade.nodes("root"))
    refs = []
    for z in zs[1:]:
        r = add_reference_node(mesh, (shear_centre[0], shear_centre[1], float(z)))
        rigid_link(mesh, bcs, r, blade.section_nodes(float(z)), dofs=(0, 1))
        bcs.fix([r], [2, 3, 4])
        refs.append(r)
    case = LoadCase("unit_torque").add_nodal_load(refs[-1], 5, 1.0)
    sol = solver.solve_many(mesh, bcs, [case])[0]
    z = zs[1:] - z0
    Lb = float(zs[-1] - z0)
    twist = np.array([sol.node_displacement(r)[5] for r in refs])  # per unit torque
    best = None
    for kL in np.geomspace(0.5, 400.0, 800):
        k = kL / Lb
        ea, eb = np.exp(-2.0 * k * (Lb - z)), np.exp(-2.0 * kL)
        sh = -np.exp(-k * z) * (1.0 - ea) / (1.0 + eb)  # sinh(k(z - L)) / cosh(kL)
        F = z - (sh + (1.0 - eb) / (1.0 + eb)) / k
        # twist = alpha z + beta F with alpha = (1 - rho) / GJ, beta = rho / GJ
        M = np.column_stack([z, F])
        coef, *_ = np.linalg.lstsq(M, twist, rcond=None)
        r = float(np.sum((M @ coef - twist) ** 2))
        if best is None or r < best[0]:
            best = (r, k, coef)
    _, k, (alpha, beta) = best
    GJ_rigid = 1.0 / (alpha + beta)
    rho = min(max(beta * GJ_rigid, 0.0), 1.0)
    if rho >= 1.0 - 1e-6:
        return float(GJ_rigid / k**2), None, float(GJ_rigid)
    if rho <= 0.0:
        raise ValueError("the shell's twist shows no warping restraint: EΓ cannot be measured on this segment")
    S = rho * GJ_rigid / (1.0 - rho)
    EGamma = GJ_rigid * S / (k**2 * (GJ_rigid + S))
    return float(EGamma), float(S), float(GJ_rigid)


__all__ = ["Airfoil", "Station", "SkinRegion", "ShearWeb", "BladeBuilder", "BladeMesh",
           "BeamSectionProperties", "beam_section_properties", "transform_beam_stiffness"]
