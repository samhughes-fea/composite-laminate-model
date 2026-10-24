"""Pluggable FEA kernels (Adapter + Strategy Patterns).

All kernels consume the same ``AnalysisModel`` and return a ``KernelResult``:

* ``NativePythonKernelAdapter``  wraps the in-house NumPy/SciPy solver pipeline.
* ``CalculiXKernelAdapter``      writes an Abaqus-style ``.inp`` deck (composite S8R/S6 shells),
                                 runs ``ccx`` and parses the ``.dat`` tables.
* ``NastranKernelAdapter``       writes a fixed-field MSC/NX Nastran bulk deck (GRID, CQUAD4,
                                 CTRIA3, CQUAD8, PCOMP, MAT8, ...), runs ``nastran`` and parses the ``.f06``.

External kernels always write their deck. When the executable is not installed
they return status ``SKIPPED`` (with the deck path) instead of raising, so a
benchmark run completes on any machine. External decks use their solver's own shell
formulation, so enriched kinematics (full RZT, full layerwise) are a native-only feature.
"""

from .base import (
    GLOBAL_COMPONENTS,
    MATERIAL_COMPONENTS,
    AnalysisModel,
    FEAKernel,
    KernelResult,
    KernelStatus,
    ProcessRunner,
    UnsupportedFeatureError,
    gauss2_heights,
)
from .calculix import CalculiXDeck, CalculiXDeckWriter, CalculiXKernelAdapter, CalculiXResultReader
from .decks import DeckCatalogue, equivalent_alphas, equivalent_nodal_forces, equivalent_thermal_load
from .nastran import NastranDeck, NastranDeckWriter, NastranF06Reader, NastranFieldFormatter, NastranKernelAdapter
from .native import NativePythonKernelAdapter
from .registry import DEFAULT_KERNELS, KernelRegistry, create_kernel

__all__ = [
    "GLOBAL_COMPONENTS", "MATERIAL_COMPONENTS", "AnalysisModel", "FEAKernel", "KernelResult", "KernelStatus",
    "ProcessRunner", "UnsupportedFeatureError", "gauss2_heights", "CalculiXDeck", "CalculiXDeckWriter",
    "CalculiXKernelAdapter", "CalculiXResultReader", "DeckCatalogue", "equivalent_alphas",
    "equivalent_nodal_forces", "equivalent_thermal_load", "NastranDeck", "NastranDeckWriter", "NastranF06Reader",
    "NastranFieldFormatter", "NastranKernelAdapter", "NativePythonKernelAdapter", "DEFAULT_KERNELS",
    "KernelRegistry", "create_kernel",
]
