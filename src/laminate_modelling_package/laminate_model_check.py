"""Pre-solve model checks: element quality, shell slenderness and a units hint.

``ModelChecker().check(mesh, bcs)`` returns a ``ModelReport`` listing what looks
wrong before any solve. Nothing here changes the model; the solver runs whatever
it is given. Thresholds are the usual rules of thumb and can be changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .laminate_kinematics import CLPTModel


@dataclass(frozen=True)
class ModelIssue:
    severity: str  # "error" or "warning"
    code: str
    message: str
    element_ids: Sequence[int] = ()


@dataclass
class ModelReport:
    issues: List[ModelIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ModelIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ModelIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set:
        return {i.code for i in self.issues}

    def __str__(self) -> str:
        if not self.issues:
            return "Model check: no issues"
        lines = [f"Model check: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for issue in self.issues:
            ids = f" (elements {list(issue.element_ids)[:8]}{'...' if len(issue.element_ids) > 8 else ''})" if issue.element_ids else ""
            lines.append(f"  [{issue.severity}] {issue.code}: {issue.message}{ids}")
        return "\n".join(lines)


class ModelChecker:
    """Rules of thumb for shell models.

    * ``aspect_ratio``: longest over shortest element edge.
    * ``warping``: out-of-plane node offset over element size.
    * ``jacobian``: min / max det J over the integration points (<= 0 is an error).
    * ``element_to_thickness``: element size below this multiple of the laminate
      thickness is outside shell theory.
    * ``clpt_slenderness``: span / thickness below which CLPT misses shear deformation.
    """

    def __init__(
        self,
        aspect_ratio: float = 5.0,
        warping: float = 0.05,
        jacobian: float = 0.2,
        element_to_thickness: float = 0.5,
        clpt_slenderness: float = 20.0,
    ):
        self.aspect_ratio = aspect_ratio
        self.warping = warping
        self.jacobian = jacobian
        self.element_to_thickness = element_to_thickness
        self.clpt_slenderness = clpt_slenderness

    def check(self, mesh, bcs=None) -> ModelReport:
        report = ModelReport()
        if not mesh.elements:
            report.issues.append(ModelIssue("error", "empty_mesh", "The mesh has no elements"))
            return report
        self._quality(mesh, report)
        self._thickness(mesh, report)
        self._units(mesh, report)
        if bcs is not None:
            self._supports(mesh, bcs, report)
        return report

    # ---------------------------------------------------------------- checks
    def _quality(self, mesh, report: ModelReport) -> None:
        bad_aspect, bad_warp, bad_jac, inverted = [], [], [], []
        for element in mesh.elements:
            q = element.quality()
            if q["jacobian_ratio"] <= 0.0:
                inverted.append(element.element_id)
            elif q["jacobian_ratio"] < self.jacobian:
                bad_jac.append(element.element_id)
            if q["aspect_ratio"] > self.aspect_ratio:
                bad_aspect.append(element.element_id)
            if q["warping_ratio"] > self.warping:
                bad_warp.append(element.element_id)
        if inverted:
            report.issues.append(ModelIssue("error", "inverted_element",
                                            "det J changes sign inside the element (inverted or badly distorted)", inverted))
        if bad_jac:
            report.issues.append(ModelIssue("warning", "jacobian_ratio",
                                            f"Jacobian ratio below {self.jacobian:g}", bad_jac))
        if bad_aspect:
            report.issues.append(ModelIssue("warning", "aspect_ratio",
                                            f"Aspect ratio above {self.aspect_ratio:g}", bad_aspect))
        if bad_warp:
            report.issues.append(ModelIssue("warning", "warping",
                                            f"Warping above {100 * self.warping:g}% of element size", bad_warp))

    def _span(self, mesh) -> float:
        xyz = np.array([mesh.node_coordinates(n) for n in mesh.node_ids])
        extent = np.ptp(xyz, axis=0)
        positive = np.sort(extent[extent > 0])
        # plate span: the smaller in-plane extent of the model's bounding box
        return float(positive[-2] if len(positive) >= 2 else positive[-1])

    def _thickness(self, mesh, report: ModelReport) -> None:
        thick, clpt = [], []
        span = self._span(mesh)
        for element in mesh.elements:
            h = element.laminate.thickness
            if element.characteristic_length < self.element_to_thickness * h:
                thick.append(element.element_id)
            if isinstance(element.kinematics, CLPTModel) and span / h < self.clpt_slenderness:
                clpt.append(element.element_id)
        if thick:
            report.issues.append(ModelIssue("warning", "thick_element",
                                            "Elements smaller than the laminate thickness; shell results "
                                            "there are not meaningful (use a solid model locally)", thick))
        if clpt:
            report.issues.append(ModelIssue("warning", "clpt_thick_plate",
                                            f"CLPT on a plate with span/thickness below {self.clpt_slenderness:g}; "
                                            "use FSDT, RZT or layerwise kinematics", clpt))

    def _units(self, mesh, report: ModelReport) -> None:
        """Flag modulus/length combinations that are unlikely to be consistent.

        Consistent sets: (m, Pa, N) gives moduli ~1e9-1e12 with spans ~1e-3-1e2;
        (mm, MPa, N) gives moduli ~1e3-1e6 with spans ~1-1e5.
        """
        E = max(max(p.material.elastic.E1, p.material.elastic.E2) for e in mesh.elements for p in e.laminate.plies)
        h = mesh.elements[0].laminate.thickness
        span = self._span(mesh)
        hint = None
        if E > 1e8 and (span > 200.0 or h > 1.0):
            hint = (f"Moduli look like Pa (max E = {E:.3g}) but the model is {span:.3g} long and {h:.3g} thick, "
                    "which looks like mm. Use metres with Pa, or mm with MPa.")
        elif E < 1e7 and (span < 0.5 and h < 0.01):
            hint = (f"Moduli look like MPa (max E = {E:.3g}) but the model is {span:.3g} long and {h:.3g} thick, "
                    "which looks like metres. Use mm with MPa, or metres with Pa.")
        if hint:
            report.issues.append(ModelIssue("warning", "units", hint))
        if h > span:
            report.issues.append(ModelIssue("error", "thickness_exceeds_span",
                                            f"Laminate thickness {h:.3g} exceeds the model span {span:.3g}"))

    def _supports(self, mesh, bcs, report: ModelReport) -> None:
        fixed = {d for (_, d) in bcs.prescribed} if bcs.prescribed else set()
        has_constraints = bool(bcs.constraints) or bool(bcs.springs)
        translational = {0, 1, 2} & {d for d in fixed if isinstance(d, int)}
        if not has_constraints and len(translational) < 3:
            missing = sorted({"UX", "UY", "UZ"} - {("UX", "UY", "UZ")[d] for d in translational})
            report.issues.append(ModelIssue("error", "unsupported_rigid_body",
                                            f"No support in {', '.join(missing)}: the stiffness matrix will be singular"))
