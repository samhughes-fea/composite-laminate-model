"""Cross-solver verification of displacements and through-thickness stress profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ThicknessProfile:
    """Stress components sampled through the thickness at one in-plane location.

    ``values`` has shape (n_points, n_components); ``z`` is measured from the mid-plane.
    ``frame`` is 'global' (shell axes) or 'material' (fibre axes).
    """

    element_id: int
    z: np.ndarray
    values: np.ndarray
    components: tuple
    frame: str = "global"
    source: str = ""

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=float).ravel()
        values = np.asarray(self.values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape != (z.size, len(self.components)):
            raise ValueError(
                f"values shape {values.shape} does not match {z.size} points x {len(self.components)} components"
            )
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "values", values)

    def component(self, name: str) -> np.ndarray:
        return self.values[:, self.components.index(name)]


@dataclass(frozen=True)
class DistributionErrorStatistics:
    """Scalar error measures between a reference and a candidate array."""

    label: str
    n_points: int
    mean_absolute_error: float
    max_absolute_error: float
    rms_error: float
    reference_peak: float
    normalized_max_error: float

    def as_row(self) -> List[str]:
        return [
            self.label,
            str(self.n_points),
            f"{self.mean_absolute_error:.4e}",
            f"{self.max_absolute_error:.4e}",
            f"{self.rms_error:.4e}",
            f"{self.normalized_max_error * 100:.2f}%",
        ]


class StressDistributionVerifier:
    """Benchmarks candidate solver outputs against a reference (normally the native engine)."""

    HEADER = ["quantity", "n", "MAE", "max abs err", "RMS", "max err / peak"]

    def compare_arrays(self, reference: Sequence[float], candidate: Sequence[float], label: str = "") -> DistributionErrorStatistics:
        ref = np.asarray(reference, dtype=float).ravel()
        cand = np.asarray(candidate, dtype=float).ravel()
        if ref.shape != cand.shape:
            raise ValueError(f"Shape mismatch for '{label}': {ref.shape} vs {cand.shape}")
        mask = np.isfinite(ref) & np.isfinite(cand)
        if not np.any(mask):
            raise ValueError(f"No finite values to compare for '{label}'")
        delta = cand[mask] - ref[mask]
        peak = float(np.max(np.abs(ref[mask])))
        max_err = float(np.max(np.abs(delta)))
        return DistributionErrorStatistics(
            label=label,
            n_points=int(mask.sum()),
            mean_absolute_error=float(np.mean(np.abs(delta))),
            max_absolute_error=max_err,
            rms_error=float(np.sqrt(np.mean(delta**2))),
            reference_peak=peak,
            normalized_max_error=max_err / peak if peak > 0.0 else (0.0 if max_err == 0.0 else float("inf")),
        )

    def profile_delta(self, reference: ThicknessProfile, candidate: ThicknessProfile, component: str) -> np.ndarray:
        """Candidate minus reference, with the candidate linearly interpolated onto reference z."""
        if reference.frame != candidate.frame:
            raise ValueError(f"Frame mismatch: {reference.frame} vs {candidate.frame}")
        z_c = candidate.z
        order = np.argsort(z_c, kind="stable")
        values = np.interp(reference.z, z_c[order], candidate.component(component)[order])
        return values - reference.component(component)

    def compare_profiles(
        self,
        reference: ThicknessProfile,
        candidate: ThicknessProfile,
        components: Optional[Iterable[str]] = None,
    ) -> Dict[str, DistributionErrorStatistics]:
        """Per-component statistics between two thickness profiles.

        If both profiles share identical z sampling the values are compared directly;
        otherwise the candidate is interpolated onto the reference heights.
        """
        names = list(components) if components is not None else [c for c in reference.components if c in candidate.components]
        stats = {}
        same_grid = reference.z.shape == candidate.z.shape and np.allclose(reference.z, candidate.z, rtol=0, atol=1e-12)
        for name in names:
            ref_values = reference.component(name)
            if same_grid:
                cand_values = candidate.component(name)
            else:
                cand_values = ref_values + self.profile_delta(reference, candidate, name)
            stats[name] = self.compare_arrays(ref_values, cand_values, f"E{reference.element_id} {name}")
        return stats

    def compare_displacements(
        self,
        reference: Mapping[int, np.ndarray],
        candidate: Mapping[int, np.ndarray],
        dofs: Sequence[int] = (0, 1, 2),
        labels: Sequence[str] = ("UX", "UY", "UZ", "RX", "RY", "RZ"),
    ) -> Dict[str, DistributionErrorStatistics]:
        """Statistics on nodal displacements for nodes present in both results."""
        common = [n for n in reference if n in candidate]
        if not common:
            raise ValueError("No common nodes between the two displacement sets")
        stats = {}
        for d in dofs:
            ref = np.array([reference[n][d] for n in common])
            cand = np.array([candidate[n][d] for n in common])
            if np.all(~np.isfinite(cand)):
                continue
            stats[labels[d]] = self.compare_arrays(ref, cand, labels[d])
        return stats

    @classmethod
    def format_table(cls, stats: Iterable[DistributionErrorStatistics], title: str = "") -> str:
        rows = [cls.HEADER] + [s.as_row() for s in stats]
        widths = [max(len(r[i]) for r in rows) for i in range(len(cls.HEADER))]
        lines = []
        if title:
            lines.append(title)
        for j, row in enumerate(rows):
            lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
            if j == 0:
                lines.append("  ".join("-" * w for w in widths))
        return "\n".join(lines)
