"""Load introduction and connections: reference nodes, rigid links (RBE2) and distributing links (RBE3).

A blade or wing is loaded through its sections: hub loads at the root, sectional
forces and moments from an aeroelastic code along the span. ``add_reference_node``
creates a node that belongs to no element; ``rigid_link`` (RBE2) makes a set of nodes
move rigidly with it, and ``distributing_link`` (RBE3) makes it follow the weighted
average motion of the set without stiffening it. Loads and supports applied to the
reference node then reach the section, and its displacements report the section's
mean translation and rotation.

Both are linear multi-point constraints in ``BoundaryConditions``, so they work with
every element type and with the CalculiX and Nastran decks (as *EQUATION / MPC).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

ALL = (0, 1, 2, 3, 4, 5)


def add_reference_node(mesh, xyz: Sequence[float], node_id: Optional[int] = None) -> int:
    """Add a node with no element (for connections); returns its id."""
    node_id = (max(mesh.node_ids) + 1 if mesh.node_ids else 1) if node_id is None else int(node_id)
    mesh.add_node(node_id, xyz)
    return node_id


def rigid_link(mesh, bcs, master: int, slaves: Iterable[int], dofs: Sequence[int] = ALL,
               rotations: bool = True) -> "BoundaryConditions":  # noqa: F821
    """RBE2: each slave node follows the master rigidly, u_s = u_m + theta_m x (x_s - x_m).

    ``dofs`` limits the tied translations (0-2) and rotations (3-5); ``rotations=False``
    leaves the slave rotations free (for solid nodes, whose rotations carry no stiffness,
    or to model a hinge line).
    """
    master = int(master)
    xm = mesh.node_coordinates(master)
    dofs = tuple(int(d) for d in dofs)
    for s in (int(n) for n in slaves):
        if s == master:
            continue
        r = mesh.node_coordinates(s) - xm
        terms = {
            0: [((master, 0), 1.0), ((master, 4), r[2]), ((master, 5), -r[1])],
            1: [((master, 1), 1.0), ((master, 5), r[0]), ((master, 3), -r[2])],
            2: [((master, 2), 1.0), ((master, 3), r[1]), ((master, 4), -r[0])],
        }
        for d in dofs:
            if d < 3:
                bcs.add_constraint((s, d), [t for t in terms[d] if t[1] != 0.0])
            elif rotations:
                bcs.add_constraint((s, d), [((master, d), 1.0)])
    return bcs


def distributing_link(mesh, bcs, reference: int, nodes: Iterable[int], weights: Optional[Sequence[float]] = None,
                      rotations: bool = True) -> "BoundaryConditions":  # noqa: F821
    """RBE3: the reference node follows the weighted mean motion of ``nodes``.

    Its translation is the weighted mean translation of the nodes, carried to the
    reference point by the mean rotation, and its rotation is the weighted
    least-squares rotation of the node cloud about its weighted centroid. The nodes keep
    their own stiffness: a load on the reference node is shared out as forces in
    proportion to the weights (and their distance from the centroid for moments), and
    an unloaded reference node is a pure measuring point. ``rotations=False`` leaves the
    reference rotations free (needed when the nodes all lie on one line).
    """
    reference = int(reference)
    nodes = [int(n) for n in nodes if int(n) != reference]
    if len(nodes) < (3 if rotations else 1):
        raise ValueError("a distributing link needs at least three nodes (one without rotations)")
    w = np.ones(len(nodes)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(nodes),) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, one per node, not all zero")
    w = w / w.sum()
    X = np.array([mesh.node_coordinates(n) for n in nodes])
    c = w @ X
    R = X - c
    # least-squares rotation: theta = A^-1 sum w_i r_i x u_i, A = sum w_i (|r|^2 I - r r^T)
    A = sum(wi * (ri @ ri * np.eye(3) - np.outer(ri, ri)) for wi, ri in zip(w, R))
    if rotations:
        if np.linalg.matrix_rank(A, tol=1e-12 * max(np.trace(A), 1e-300)) < 3:
            raise ValueError("the nodes are collinear: use rotations=False")
        Ainv = np.linalg.inv(A)
        # theta = sum_i C_i u_i with C_i = w_i A^-1 [r_i]x
        C = [wi * Ainv @ _skew(ri) for wi, ri in zip(w, R)]
    else:
        C = None
    d = mesh.node_coordinates(reference) - c
    for k in range(3):
        # u_ref = sum w_i u_i + theta x d = sum (w_i I - [d]x C_i) u_i
        masters = []
        for i, n in enumerate(nodes):
            row = w[i] * np.eye(3)[k]
            if C is not None:
                row = row - (_skew(d) @ C[i])[k]
            masters += [((n, j), float(row[j])) for j in range(3) if row[j] != 0.0]
        bcs.add_constraint((reference, k), masters)
    if C is not None:
        for k in range(3):
            masters = [((n, j), float(C[i][k, j])) for i, n in enumerate(nodes) for j in range(3) if C[i][k, j] != 0.0]
            bcs.add_constraint((reference, 3 + k), masters)
    return bcs


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


__all__ = ["add_reference_node", "rigid_link", "distributing_link"]
