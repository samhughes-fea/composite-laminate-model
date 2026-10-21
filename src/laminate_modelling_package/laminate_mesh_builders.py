"""Structured mesh builders for plates and cylindrical panels.

Builders create the nodes and elements of simple geometries and record named node
sets for the edges, so boundary conditions and line loads can be applied by name.
General geometry is meshed with Gmsh (``laminate_gmsh``).

``element`` selects the element type: ``"quad4"`` (default), ``"tri3"`` (each cell
split along its diagonal) or ``"quad8"`` (mid-side nodes added).
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


def _structured(
    nu: int,
    nv: int,
    point: Callable[[float, float], np.ndarray],
    laminate: Laminate,
    kinematics: KinematicModel,
    element: str,
    edge_names: Tuple[str, str, str, str],
    element_options: dict,
) -> NamedMesh:
    """Mesh the parameter square [0, 1]^2 mapped by ``point(s, t)``; edges named (s=0, s=1, t=0, t=1)."""
    from .laminate_solver_pipeline import FEAMesh

    if nu < 1 or nv < 1:
        raise ValueError("the number of elements in each direction must be >= 1")
    if element not in ("quad4", "tri3", "quad8"):
        raise ValueError("element must be 'quad4', 'tri3' or 'quad8'")
    mesh = FEAMesh()
    step = 2 if element == "quad8" else 1
    grid: Dict[Tuple[int, int], int] = {}
    node = 1
    for j in range(step * nv + 1):
        for i in range(step * nu + 1):
            if step == 2 and i % 2 == 1 and j % 2 == 1:
                continue  # serendipity: no centre node
            mesh.add_node(node, point(i / (step * nu), j / (step * nv)))
            grid[(i, j)] = node
            node += 1
    eid = 1
    for j in range(nv):
        for i in range(nu):
            if element == "quad8":
                a, b = 2 * i, 2 * j
                conn = [grid[(a, b)], grid[(a + 2, b)], grid[(a + 2, b + 2)], grid[(a, b + 2)],
                        grid[(a + 1, b)], grid[(a + 2, b + 1)], grid[(a + 1, b + 2)], grid[(a, b + 1)]]
                mesh.add_element(eid, conn, laminate, kinematics, **element_options)
                eid += 1
                continue
            n1, n2, n3, n4 = grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]
            if element == "quad4":
                mesh.add_element(eid, (n1, n2, n3, n4), laminate, kinematics, **element_options)
                eid += 1
            else:
                mesh.add_element(eid, (n1, n2, n3), laminate, kinematics, **element_options)
                mesh.add_element(eid + 1, (n1, n3, n4), laminate, kinematics, **element_options)
                eid += 2
    imax, jmax = step * nu, step * nv
    sets = {
        edge_names[0]: [n for (i, j), n in grid.items() if i == 0],
        edge_names[1]: [n for (i, j), n in grid.items() if i == imax],
        edge_names[2]: [n for (i, j), n in grid.items() if j == 0],
        edge_names[3]: [n for (i, j), n in grid.items() if j == jmax],
    }
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
    element: str = "quad4"

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
    element: str = "quad4"

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
        from .laminate_solver_pipeline import FEAMesh

        if self.element != "quad4":
            raise ValueError("closed barrels are built with quad4 elements")
        mesh = FEAMesh()
        n_c, n_a = self.n_hoop, self.n_axial
        ids = {}
        node = 1
        for j in range(n_a + 1):
            for i in range(n_c):
                phi = np.radians(self.phi_start + 360.0 * i / n_c)
                mesh.add_node(node, (j * self.length / n_a, self.radius * np.cos(phi), self.radius * np.sin(phi)))
                ids[(i, j)] = node
                node += 1
        eid = 1
        for j in range(n_a):
            for i in range(n_c):
                k = (i + 1) % n_c
                mesh.add_element(eid, (ids[(i, j)], ids[(k, j)], ids[(k, j + 1)], ids[(i, j + 1)]),
                                 laminate, kinematics, **element_options)
                eid += 1
        sets = {"x0": [ids[(i, 0)] for i in range(n_c)], "x1": [ids[(i, n_a)] for i in range(n_c)], "all": list(ids.values())}
        return NamedMesh(mesh, sets, {"all": mesh.element_ids})

    def build(self, laminate: Laminate, kinematics: KinematicModel, **element_options):
        return self.build_named(laminate, kinematics, **element_options).mesh
