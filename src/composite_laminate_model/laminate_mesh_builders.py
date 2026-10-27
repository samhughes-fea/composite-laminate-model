"""Structured mesh builders for plates and cylindrical panels.

Builders create the nodes and elements of simple geometries and record named node
sets for the edges, so boundary conditions and line loads can be applied by name.
General geometry is meshed with Gmsh (``laminate_gmsh``).

``element`` selects the element type: ``"quad8_curved"`` (curved 8-node shell, the
default), ``"quad4"``, ``"tri3"`` (each cell split along its diagonal), ``"quad8"``
(flat 8-node) or ``"tri6"`` (curved 6-node triangles). Quadratic elements put their
mid-side nodes on the exact geometry, so the curved elements follow it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .laminate_kinematics import KinematicModel
from .laminate_mechanics import Laminate


@dataclass
class NamedMesh:
    """A mesh with named node and element sets."""

    mesh: "FEAMesh"  # noqa: F821
    node_sets: Dict[str, List[int]] = field(default_factory=dict)
    element_sets: Dict[str, List[int]] = field(default_factory=dict)

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


ELEMENT_CHOICES = ("quad4", "tri3", "quad8", "quad8_curved", "tri6")


def _structured(
    nu: int,
    nv: int,
    point: Callable[[float, float], np.ndarray],
    laminate: Laminate,
    kinematics: KinematicModel,
    element: str,
    edge_names: Tuple[str, str, str, str],
    element_options: dict,
    periodic_u: bool = False,
) -> NamedMesh:
    """Mesh the parameter square [0, 1]^2 mapped by ``point(s, t)``; edges named (s=0, s=1, t=0, t=1).

    ``periodic_u`` closes the mesh in s (s = 1 is s = 0), for full barrels; the s
    edges then have no names.
    """
    from .laminate_solver_pipeline import FEAMesh

    if nu < 1 or nv < 1:
        raise ValueError("the number of elements in each direction must be >= 1")
    if element not in ELEMENT_CHOICES:
        raise ValueError(f"element must be one of {', '.join(ELEMENT_CHOICES)}")
    if periodic_u and nu < 3:
        raise ValueError("a closed mesh needs at least 3 elements around")
    mesh = FEAMesh()
    quadratic = element in ("quad8", "quad8_curved", "tri6")
    step = 2 if quadratic else 1
    imax, jmax = step * nu, step * nv
    grid: Dict[Tuple[int, int], int] = {}
    node = 1
    for j in range(jmax + 1):
        for i in range(imax + (0 if periodic_u else 1)):
            if element in ("quad8", "quad8_curved") and i % 2 == 1 and j % 2 == 1:
                continue  # serendipity: no centre node
            mesh.add_node(node, point(i / imax, j / jmax))
            grid[(i, j)] = node
            node += 1

    def g(i, j):
        return grid[(i % imax if periodic_u else i, j)]

    element_type = "quad8_curved" if element == "quad8_curved" else None
    eid = 1
    for j in range(nv):
        for i in range(nu):
            if quadratic:
                a, b = 2 * i, 2 * j
                if element == "tri6":
                    mesh.add_element(eid, [g(a, b), g(a + 2, b), g(a + 2, b + 2), g(a + 1, b), g(a + 2, b + 1),
                                           g(a + 1, b + 1)], laminate, kinematics, **element_options)
                    mesh.add_element(eid + 1, [g(a, b), g(a + 2, b + 2), g(a, b + 2), g(a + 1, b + 1),
                                               g(a + 1, b + 2), g(a, b + 1)], laminate, kinematics, **element_options)
                    eid += 2
                else:
                    conn = [g(a, b), g(a + 2, b), g(a + 2, b + 2), g(a, b + 2),
                            g(a + 1, b), g(a + 2, b + 1), g(a + 1, b + 2), g(a, b + 1)]
                    mesh.add_element(eid, conn, laminate, kinematics, element_type=element_type, **element_options)
                    eid += 1
                continue
            n1, n2, n3, n4 = g(i, j), g(i + 1, j), g(i + 1, j + 1), g(i, j + 1)
            if element == "quad4":
                mesh.add_element(eid, (n1, n2, n3, n4), laminate, kinematics, **element_options)
                eid += 1
            else:
                mesh.add_element(eid, (n1, n2, n3), laminate, kinematics, **element_options)
                mesh.add_element(eid + 1, (n1, n3, n4), laminate, kinematics, **element_options)
                eid += 2
    sets = {}
    if not periodic_u:
        sets[edge_names[0]] = [n for (i, j), n in grid.items() if i == 0]
        sets[edge_names[1]] = [n for (i, j), n in grid.items() if i == imax]
    sets[edge_names[2]] = [n for (i, j), n in grid.items() if j == 0]
    sets[edge_names[3]] = [n for (i, j), n in grid.items() if j == jmax]
    sets["all"] = list(grid.values())
    return NamedMesh(mesh, sets, {"all": mesh.element_ids})


@dataclass
class RectangularPlateBuilder:
    """nx-by-ny plate in the plane z = origin_z; node ids start at 1, row by row.

    Edge node sets: ``left`` (x = x0), ``right``, ``bottom`` (y = y0) and ``top``.
    """

    length_x: float
    length_y: float
    nx: int
    ny: int
    origin: Sequence[float] = (0.0, 0.0, 0.0)
    element: str = "quad8_curved"

    def build_named(self, laminate: Laminate, kinematics: KinematicModel, **element_options) -> NamedMesh:
        x0, y0, z0 = (list(self.origin) + [0.0, 0.0, 0.0])[:3]
        point = lambda s, t: np.array([x0 + s * self.length_x, y0 + t * self.length_y, z0])  # noqa: E731
        return _structured(self.nx, self.ny, point, laminate, kinematics, self.element,
                           ("left", "right", "bottom", "top"), element_options)

    def build(self, laminate: Laminate, kinematics: KinematicModel, **element_options):
        return self.build_named(laminate, kinematics, **element_options).mesh


@dataclass
class CylindricalPanelBuilder:
    """Cylindrical panel (or full barrel) about the global x axis.

    Nodes sit at (x, R cos(phi), R sin(phi)) for x in [0, length] and phi in
    [phi_start, phi_end] (degrees). Element normals point outward. ``n_hoop`` elements
    span the arc and ``n_axial`` the length. A full 360-degree span closes the seam.

    Node sets: ``x0`` and ``x1`` (the two ends) and, for open panels, ``phi0`` and
    ``phi1`` (the straight edges).
    """

    radius: float
    length: float
    n_axial: int
    n_hoop: int
    phi_start: float = 0.0
    phi_end: float = 90.0
    element: str = "quad8_curved"

    def build_named(self, laminate: Laminate, kinematics: KinematicModel, **element_options) -> NamedMesh:
        span = self.phi_end - self.phi_start
        closed = abs(abs(span) - 360.0) < 1e-9
        if closed:
            return self._closed(laminate, kinematics, element_options)

        def point(s, t):
            phi = np.radians(self.phi_start + s * span)
            return np.array([t * self.length, self.radius * np.cos(phi), self.radius * np.sin(phi)])

        named = _structured(self.n_hoop, self.n_axial, point, laminate, kinematics, self.element,
                            ("phi0", "phi1", "x0", "x1"), element_options)
        return named

    def _closed(self, laminate, kinematics, element_options) -> NamedMesh:
        def point(s, t):
            phi = np.radians(self.phi_start + 360.0 * s)
            return np.array([t * self.length, self.radius * np.cos(phi), self.radius * np.sin(phi)])

        return _structured(self.n_hoop, self.n_axial, point, laminate, kinematics, self.element,
                           ("phi0", "phi1", "x0", "x1"), element_options, periodic_u=True)

    def build(self, laminate: Laminate, kinematics: KinematicModel, **element_options):
        return self.build_named(laminate, kinematics, **element_options).mesh
