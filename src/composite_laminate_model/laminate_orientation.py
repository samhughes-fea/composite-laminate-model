"""Material orientation: the direction that ply angles are measured from.

Each element's local x axis (the 0-degree fibre direction) is the projection of a
reference direction onto the element plane. A direction can be a plain 3-vector or
any object with ``direction_at(xyz)``; the classes here cover the usual cases.

* ``FixedDirection``        one global vector (the default is global x).
* ``CylindricalDirection``  axial or hoop direction of a cylinder (tubes, spars, fuselage barrels).
* ``FieldDirection``        any function of position (blade sections, draped layups).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class FixedDirection:
    vector: Sequence[float] = (1.0, 0.0, 0.0)

    def direction_at(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(self.vector, dtype=float)


@dataclass(frozen=True)
class CylindricalDirection:
    """Axial or hoop direction of a cylinder with the given axis.

    ``reference="axial"`` measures ply angles from the cylinder axis;
    ``reference="hoop"`` from the circumferential direction (right-handed about the axis).
    """

    axis_point: Sequence[float] = (0.0, 0.0, 0.0)
    axis_direction: Sequence[float] = (1.0, 0.0, 0.0)
    reference: str = "axial"

    def __post_init__(self):
        if self.reference not in ("axial", "hoop"):
            raise ValueError("reference must be 'axial' or 'hoop'")

    def direction_at(self, xyz: np.ndarray) -> np.ndarray:
        a = np.asarray(self.axis_direction, dtype=float)
        a = a / np.linalg.norm(a)
        if self.reference == "axial":
            return a
        r = np.asarray(xyz, dtype=float) - np.asarray(self.axis_point, dtype=float)
        r = r - np.dot(r, a) * a
        if np.linalg.norm(r) < 1e-300:
            raise ValueError("The hoop direction is undefined on the cylinder axis")
        return np.cross(a, r / np.linalg.norm(r))


@dataclass(frozen=True)
class FieldDirection:
    """Reference direction given by a function of the global position."""

    function: Callable[[np.ndarray], Sequence[float]]

    def direction_at(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(self.function(np.asarray(xyz, dtype=float)), dtype=float)
