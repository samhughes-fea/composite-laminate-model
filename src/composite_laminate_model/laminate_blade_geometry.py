"""Shell models of a blade meshed from its implicit geometry (``implicit_geometry_kernel``).

``GeometryBladeBuilder`` (made by ``BladeBuilder.from_geometry``) cuts a
``BladeGeometry`` at every node section along the span and meshes the walls'
mid-surfaces between the cuts. Any of the geometry's structural layouts can be meshed:
skins, spar caps, any number of shear webs, trailing-edge webs and boxes.

The shells lie where ``SectionCut.mid_surface()`` puts the walls, the same graph
``ThinWalledSection.from_geometry`` builds its section from:

- skins on their mid-surface (inward from the outer mould line by half their
  thickness), with normals out of the section;
- spar caps merged into the skin they are bonded to: along the bond the shell is the
  cap and skin stacked (cap plies inside, skin plies outside) on the stack's
  mid-surface, as a cap laid on a skin is;
- webs between the skins' mid-surfaces, with normals towards the trailing edge.

``caps="tied"`` keeps the spar caps as shells of their own instead, as a cap bonded to
a skin is built: the skin runs on its own mid-surface under the cap, the cap lies on
its own mid-surface inside it (half the skin and half the cap thickness in from the
skin's), and each cap node is tied to the skin node facing it by a rigid offset link
(RBE2). Webs then end on the skins' mid-surfaces, and a cap crossing a web is joined to
it only through the skin. The links are ``BladeMesh.ties``; add them to the boundary
conditions with ``blade.tie_caps(bcs)`` (an untied cap floats, and the solver stops on a
singular stiffness). The cap nodes are left out of the ``root`` and ``tip`` node sets and
of ``section_nodes``, since they follow the skin through the links, so supports and
section links on those sets reach the caps without tying a node twice. This is the
model blade-structure's section shell solver uses (its ``cap_tie_mode="skin_offset"``):
each cap carries its own stresses, ply drops and element sets, and its edges end on
the skin rather than stepping the skin's reference surface. ``ThinWalledSection.from_blade_mesh``
(``anisotropic_section_model``) cannot read the links; use ``from_geometry`` there.

Each wall run between junctions (where walls meet, or where a skin changes region or
gains a cap) is one strip of elements along the span, with the same number of elements
at every section, spaced equally along the wall. The section's topology must be the
same along the whole span: a web dropped part way along (fixed ``web_offsets`` on a
tapered blade) needs separate meshes either side.

Laminates are bound by wall label, then region, then kind, as in
``ThinWalledSection.from_geometry``: ``{"skin": ..., "cap": ..., "web": ...}`` or
``{"skin_suction_surface:bay_0": ..., "shear_web_1": ...}``. A value may be a
``Laminate`` or a function of z returning one, for walls whose thickness changes along
the span. Skin and cap laminates must match the geometry's thickness (within
``thickness_tol``), because the geometry places the walls.

Element sets: every wall label and region an element belongs to
(``spar_cap_suction_surface``, ``skin_suction_surface:bay_0``,
``skin_suction_surface``, ``shear_web_0``, ...), ``skin`` (skins, including the capped
parts), ``caps``, ``webs`` and ``all``. With ``caps="tied"`` the cap elements are in
``caps`` and their cap's sets only, and the skin elements under them in the skin's.
Node sets: ``root``, ``tip``, one per web, ``all``, and ``caps`` (the tied cap nodes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .laminate_kinematics import KinematicModel
from .laminate_mechanics import Laminate


@dataclass(frozen=True)
class _Edge:
    """A run of mid-surface segments between two junctions, the same at every section."""

    key: Tuple[str, Tuple[str, ...], int]  # (name, layers, occurrence)
    kind: str
    start: int  # junction ids
    end: int


def _edges(mid) -> List[Tuple[str, str, Tuple[str, ...], List[int]]]:
    """Split the mid-surface graph into runs between junctions: [(name, kind, layers, point ids)]."""
    segs = mid.segments
    degree: Dict[int, int] = {}
    outs: Dict[int, List[int]] = {}
    ins: Dict[int, List[int]] = {}
    for k, s in enumerate(segs):
        degree[s.start] = degree.get(s.start, 0) + 1
        degree[s.end] = degree.get(s.end, 0) + 1
        outs.setdefault(s.start, []).append(k)
        ins.setdefault(s.end, []).append(k)

    def through(p):
        """The segment that carries a run on through point p, if p is not a junction."""
        if degree.get(p) != 2 or len(outs.get(p, ())) != 1 or len(ins.get(p, ())) != 1:
            return None
        a, b = segs[ins[p][0]], segs[outs[p][0]]
        if (a.name, a.kind, a.layers) != (b.name, b.kind, b.layers):
            return None
        return outs[p][0]

    junction = {p for p in degree if through(p) is None}
    if not junction:  # one closed loop of one wall: cut it at its first point
        junction = {segs[0].start}
    seen = set()
    runs = []
    for k, s in enumerate(segs):
        if k in seen or s.start not in junction:
            continue
        pts = [s.start]
        while True:
            seen.add(k)
            pts.append(segs[k].end)
            if segs[k].end in junction:
                break
            k = through(segs[k].end)
        runs.append((s.name, s.kind, s.layers, pts))
    if len(seen) != len(segs):
        raise ValueError("the section's walls do not form runs between junctions")
    return runs


def _resample(points: np.ndarray, m: int, along: Optional[np.ndarray] = None) -> np.ndarray:
    """m + 1 points equally spaced along a polyline, ends included.

    With ``along`` (a polyline with the same number of vertices), the points are spaced
    equally along ``along`` instead and placed at the same vertex parameter on
    ``points``: a cap's nodes then face the skin nodes they are tied to.
    """
    ref = points if along is None else along
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))])
    u = np.linspace(0.0, arc[-1], m + 1)
    v = np.interp(u, arc, np.arange(len(ref), dtype=float))
    i = np.minimum(np.floor(v).astype(int), len(ref) - 2)
    t = (v - i)[:, None]
    return points[i] * (1.0 - t) + points[i + 1] * t


def _cap_positions(cut, mid):
    """Where the skins and caps lie when the caps are not merged into the skins.

    Returns (skin points, cap points): the merged mid-surface's points with the skins
    back on their own mid-surfaces, and, for every point of a capped run, the cap's
    mid-surface point facing it (``None`` elsewhere). A merged point lies a fraction
    t_cap / (t_cap + t_skin) of the way from the skin point to the cap point.
    """
    from dataclasses import replace

    bare = replace(cut, walls=tuple(w for w in cut.walls if w.kind != "cap")).mid_surface()
    if bare.points.shape != mid.points.shape:
        raise RuntimeError("the geometry's mid-surface with and without caps numbers its points differently")
    skin = bare.points
    cap = np.full_like(skin, np.nan)
    for seg in mid.segments:
        if len(seg.layers) != 2:
            continue
        tc, ts = cut.wall(seg.layers[0]).thickness, cut.wall(seg.layers[1]).thickness
        f = tc / (tc + ts)
        for k in (seg.start, seg.end):
            cap[k] = skin[k] + (mid.points[k] - skin[k]) / f
    moved = np.linalg.norm(mid.points - skin, axis=1) > 0.0
    if np.any(moved & np.isnan(cap[:, 0])):
        raise RuntimeError("the geometry moved a mid-surface point that no capped run holds")
    return skin, cap


class GeometryBladeBuilder:
    """Shell model of a ``BladeGeometry``: its walls' mid-surfaces meshed along the span.

    Parameters
    ----------
    geometry:
        A ``implicit_geometry_kernel.BladeGeometry``.
    n_span:
        Elements along the span (an int), or increasing span fractions 0..1.
    element:
        ``"quad8_curved"`` (default), ``"quad8"`` or ``"quad4"``.
    element_size:
        Target element length around the section (default 1/40 of the largest chord);
        each wall run gets ceil(length / element_size) elements, at least one, from its
        longest cut.
    wall_elements:
        Element counts that override ``element_size``, by wall label, region or kind
        (``{"web": 6, "spar_cap_suction_surface": 12}``). A run along a capped skin is
        looked up by its cap first.
    material_axis:
        The 0-degree ply direction, projected onto each element (the span axis by
        default).
    caps:
        ``"merged"`` (default) stacks each spar cap with its skin on one shell;
        ``"tied"`` meshes it as a shell of its own on its mid-surface, tied to the skin
        by rigid offset links (see the module notes).
    """

    def __init__(self, geometry, n_span: Union[int, Sequence[float]] = 10, element: str = "quad8_curved",
                 element_size: Optional[float] = None, wall_elements: Optional[Dict[str, int]] = None,
                 material_axis=(0.0, 0.0, 1.0), caps: str = "merged"):
        if element not in ("quad4", "quad8", "quad8_curved"):
            raise ValueError("element must be 'quad4', 'quad8' or 'quad8_curved'")
        if caps not in ("merged", "tied"):
            raise ValueError("caps must be 'merged' or 'tied'")
        self.caps = caps
        self.geometry = geometry
        self.element = element
        self.material_axis = material_axis
        self.wall_elements = dict(wall_elements or {})
        chord = max(s.chord for s in geometry.stations)
        self.element_size = float(element_size) if element_size is not None else chord / 40.0
        if self.element_size <= 0.0:
            raise ValueError("element_size must be > 0")
        if np.ndim(n_span) == 0:
            self.span_fractions = np.linspace(0.0, 1.0, int(n_span) + 1)
        else:
            self.span_fractions = np.asarray(n_span, dtype=float)
        f = self.span_fractions
        if f.size < 2 or abs(f[0]) > 1e-12 or abs(f[-1] - 1.0) > 1e-12 or np.any(np.diff(f) <= 0):
            raise ValueError("n_span must be >= 1 or increasing fractions from 0 to 1")
        self._cuts: Dict[float, object] = {}

    # ------------------------------------------------------------ geometry
    @property
    def z_nodes(self) -> np.ndarray:
        """Span positions of the node sections (element-edge sections, and mid-sections if quadratic)."""
        sf = self.span_fractions
        if self.element != "quad4":
            sf = np.concatenate([[sf[0]], np.column_stack([0.5 * (sf[:-1] + sf[1:]), sf[1:]]).ravel()])
        z0, z1 = self.geometry.stations[0].z, self.geometry.stations[-1].z
        return z0 + sf * (z1 - z0)

    def cut(self, z: float):
        """The geometry's ``SectionCut`` at z (cached)."""
        key = round(float(z), 12)
        if key not in self._cuts:
            self._cuts[key] = self.geometry.section_at(float(z))
        return self._cuts[key]

    def _count(self, cut, layers: Tuple[str, ...], length: float) -> int:
        for label in layers:
            for key in cut.wall(label).lookup_keys:
                if key in self.wall_elements:
                    return max(1, int(self.wall_elements[key]))
        return max(1, int(np.ceil(length / self.element_size - 1e-9)))

    # ---------------------------------------------------------------- build
    def build(self, laminates, kinematics: KinematicModel, thickness_tol: float = 0.01, **element_options):
        """Mesh the blade. ``laminates`` maps wall labels, regions or kinds to laminates (or functions of z)."""
        from .laminate_blade import BladeMesh
        from .laminate_solver_pipeline import FEAMesh

        element_options = dict(element_options)
        element_options.setdefault("material_axis", self.material_axis)
        quadratic = self.element != "quad4"
        step = 2 if quadratic else 1
        etype = "quad8_curved" if self.element == "quad8_curved" else None
        zs = self.z_nodes
        cuts = [self.cut(z) for z in zs]
        mids = [c.mid_surface() for c in cuts]
        tied = self.caps == "tied"
        if tied:
            places = [_cap_positions(c, m) for c, m in zip(cuts, mids)]
        else:
            places = [(m.points, None) for m in mids]

        # topology from the first section, checked at every other one
        layout = None
        per_section = []
        for z, mid, (skin_xy, cap_xy) in zip(zs, mids, places):
            runs = _edges(mid)
            count: Dict[Tuple[str, Tuple[str, ...]], int] = {}
            keyed = []
            for name, kind, layers, pts in runs:
                n = count.get((name, layers), 0)
                count[(name, layers)] = n + 1
                keyed.append(((name, layers, n), kind, pts))
            # junctions numbered in order of the sorted runs, so the same at every section
            jid: Dict[int, int] = {}
            edges = []
            for key, kind, pts in sorted(keyed, key=lambda r: r[0]):
                for p in (pts[0], pts[-1]):
                    jid.setdefault(p, len(jid))
                capped = cap_xy is not None and len(key[1]) > 1
                edges.append((_Edge(key, kind, jid[pts[0]], jid[pts[-1]]), skin_xy[pts],
                              cap_xy[pts] if capped else None))
            topo = sorted((e.key, e.kind, e.start, e.end) for e, _, _ in edges)
            if layout is None:
                layout = topo
            elif topo != layout:
                raise ValueError(f"the section at z = {z:g} has different walls from the one at z = {zs[0]:g} "
                                 f"(a web dropped or added?); mesh the two parts separately")
            per_section.append({e.key: (e, xy, cxy) for e, xy, cxy in edges})
        keys = [k for k, _, _, _ in layout]
        counts = {}
        for key in keys:
            longest = max(max(float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
                              for xy in sec[key][1:] if xy is not None) for sec in per_section)
            counts[key] = self._count(cuts[0], key[1], longest) * step

        # nodes
        mesh = FEAMesh()
        node = 1
        junction_node: Dict[Tuple[int, int], int] = {}
        grid: Dict[Tuple[Tuple, int, int], int] = {}
        cap_grid: Dict[Tuple[Tuple, int, int], int] = {}
        ties: Dict[int, int] = {}  # cap node -> skin node it is tied to
        for j, z in enumerate(zs):
            sec = per_section[j]
            for key in keys:
                edge, xy, cxy = sec[key]
                m = counts[key]
                pts = _resample(xy, m)
                if cxy is not None:
                    cpts = _resample(cxy, m, along=xy)
                for k in range(m + 1):
                    if quadratic and k % 2 and j % 2:
                        continue
                    if k in (0, m):
                        jkey = (edge.start if k == 0 else edge.end, j)
                        if jkey not in junction_node:
                            mesh.add_node(node, (pts[k, 0], pts[k, 1], float(z)))
                            junction_node[jkey] = node
                            node += 1
                        grid[(key, k, j)] = junction_node[jkey]
                    else:
                        mesh.add_node(node, (pts[k, 0], pts[k, 1], float(z)))
                        grid[(key, k, j)] = node
                        node += 1
                    if cxy is not None:
                        mesh.add_node(node, (cpts[k, 0], cpts[k, 1], float(z)))
                        cap_grid[(key, k, j)] = node
                        ties[node] = grid[(key, k, j)]
                        node += 1

        # laminates
        def laminate_of(cut, label, z):
            wall = cut.wall(label)
            for key in wall.lookup_keys:
                if hasattr(laminates, "get") and key in laminates:
                    lam = laminates[key]
                    break
            else:
                if isinstance(laminates, Laminate):
                    lam = laminates
                else:
                    raise KeyError(f"no laminate for wall '{label}' (tried {', '.join(map(repr, wall.lookup_keys))})")
            if callable(lam) and not isinstance(lam, Laminate):
                lam = lam(z)
            if wall.kind != "web" and abs(lam.thickness - wall.thickness) > thickness_tol * wall.thickness:
                raise ValueError(f"wall '{label}' is {wall.thickness * 1e3:.4g} mm thick at z = {z:g} in the geometry "
                                 f"but its laminate is {lam.thickness * 1e3:.4g} mm")
            return lam

        stacks: Dict[Tuple, Laminate] = {}

        def stack(layers, j0, j1):
            """The laminate of a run between node sections j0 and j1 (checked at its middle section)."""
            jm = (j0 + j1) // 2 if quadratic else j0
            z = 0.5 * (zs[j0] + zs[j1])
            parts = tuple(laminate_of(cuts[jm], label, z) for label in layers)
            if len(parts) == 1:
                return parts[0]
            ids = tuple(id(p) for p in parts)
            if ids not in stacks:
                stacks[ids] = Laminate([ply for p in parts for ply in p.plies], "+".join(p.name for p in parts))
            return stacks[ids]

        # elements
        elements: Dict[str, List[int]] = {}
        web_nodes: Dict[str, List[int]] = {}
        eid = 1
        nz = len(self.span_fractions) - 1
        kinds = {k: kind for k, kind, _, _ in layout}
        def sets_of(name, layers):
            return list(dict.fromkeys([name] + [v for label in layers for v in (label, label.split(":")[0])]))

        for key in keys:
            name, layers, _ = key
            m = counts[key]
            separate = tied and len(layers) > 1
            # (node grid, wall layers, element sets) of each shell along this run
            shells = []
            if kinds[key] == "web":
                shells.append((grid, layers, sets_of(name, layers) + ["webs"]))
            elif separate:
                shells.append((grid, layers[1:], sets_of(layers[1], layers[1:]) + ["skin"]))
                shells.append((cap_grid, layers[:1], sets_of(name, layers[:1]) + ["caps"]))
            else:
                shells.append((grid, layers, sets_of(name, layers) + ["skin"] + (["caps"] if len(layers) > 1 else [])))
            for nodes_of, walls, sets in shells:
                g = lambda a, b: nodes_of[(key, a, b)]  # noqa: E731
                for jj in range(nz):
                    j, d = step * jj, step
                    lam = stack(walls, j, j + d)
                    for kk in range(0, m, step):
                        # corners along +z first, then along the wall: the normal is z x (wall direction)
                        conn = [g(kk, j), g(kk, j + d), g(kk + d, j + d), g(kk + d, j)]
                        if quadratic:
                            conn += [g(kk, j + 1), g(kk + 1, j + d), g(kk + d, j + 1), g(kk + 1, j)]
                        mesh.add_element(eid, conn, lam, kinematics, element_type=etype, **element_options)
                        for sname in sets:
                            elements.setdefault(sname, []).append(eid)
                        eid += 1
            if kinds[key] == "web":
                web_nodes.setdefault(name, [])
                web_nodes[name].extend(v for (k2, _, _), v in grid.items() if k2 == key)

        all_nodes = mesh.node_ids
        z0, z1 = float(zs[0]), float(zs[-1])
        tol = 1e-12 * max(1.0, abs(z0), abs(z1))
        node_sets: Dict[str, List[int]] = {
            "root": [n for n in all_nodes if abs(mesh.node_coordinates(n)[2] - z0) < tol and n not in ties],
            "tip": [n for n in all_nodes if abs(mesh.node_coordinates(n)[2] - z1) < tol and n not in ties],
            "all": all_nodes,
        }
        if tied:
            node_sets["caps"] = sorted(ties)
        node_sets.update({k: list(dict.fromkeys(v)) for k, v in web_nodes.items()})
        elements["all"] = mesh.element_ids
        sf = self.span_fractions
        return BladeMesh(mesh, node_sets, elements, tuple(self.geometry.stations), z0 + sf * (z1 - z0),
                         geometry=self.geometry, ties=ties)

    def __repr__(self) -> str:
        return (f"GeometryBladeBuilder({self.geometry!r}, {len(self.span_fractions) - 1} elements along the span, "
                f"{self.element}, caps {self.caps})")


__all__ = ["GeometryBladeBuilder"]
