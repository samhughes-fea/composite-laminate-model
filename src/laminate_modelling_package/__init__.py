"""laminate_modelling_package - generalised composite laminate mechanics suite.

Public API overview
-------------------
Materials       ElasticProperties3D, StrengthProperties3D, MaterialCard, MaterialLibrary
Geometry        Ply, Laminate
Environment     HygrothermalProperties, HygrothermalState, HygrothermalEngine
Kinematics      CLPTModel, FSDTModel, RefinedZigZagModel, LayerwiseModel
Element         CompositePlateElement (Quad4, 24 DOF)
Solution        FEAMesh, BoundaryConditions, LoadCase, StructuralGlobalSolver
Results         ResultsExtractor (Gauss point, element, node, ply and global levels)
Post-processing ExactStressRecoverer, failure criteria, SmearedCrackDegradation,
                ProgressiveDamageAnalysis
Kernels         NativePythonKernelAdapter, CalculiXKernelAdapter, NastranKernelAdapter
Verification    StressDistributionVerifier
"""

from .laminate_benchmark_stress import DistributionErrorStatistics, StressDistributionVerifier, ThicknessProfile
from .laminate_failure_criteria import (
    FailureCriterion,
    FailureMode,
    FailureResult,
    HashinCriterion,
    MaximumStressCriterion,
    TsaiHillCriterion,
    TsaiWuCriterion,
    create_criterion,
)
from .laminate_fea_engine import (
    CLPTModel,
    CompositePlateElement,
    FSDTModel,
    KinematicModel,
    LayerwiseModel,
    LayerwiseThicknessMatrices,
    RefinedZigZagModel,
    SectionStiffness,
    integrate_abd,
    shape_functions,
)
from .laminate_hygrothermal import HygrothermalEngine, HygrothermalProperties, HygrothermalState
from .laminate_kernels import (
    AnalysisModel,
    CalculiXDeckWriter,
    CalculiXKernelAdapter,
    CalculiXResultReader,
    FEAKernel,
    KernelRegistry,
    KernelResult,
    KernelStatus,
    NastranDeckWriter,
    NastranF06Reader,
    NastranFieldFormatter,
    NastranKernelAdapter,
    NativePythonKernelAdapter,
    ProcessRunner,
    UnsupportedFeatureError,
    create_kernel,
)
from .laminate_materials import (
    DuplicateMaterialError,
    MaterialCard,
    MaterialLibrary,
    MaterialNotFoundError,
    MaterialRepository,
    create_default_library,
)
from .laminate_materials_3d import (
    ElasticProperties3D,
    StrengthProperties3D,
    rotate_stiffness_3d,
    strain_rotation_3d,
    stress_rotation_3d,
)
from .laminate_mechanics import (
    Laminate,
    Ply,
    q_bar_from_invariants,
    reuter_matrix,
    shear_transformation,
    stiffness_invariants,
    strain_transformation,
    stress_transformation,
    transform_reduced_stiffness,
)
from .laminate_progressive_damage import (
    DamageState,
    DegradationStrategy,
    ProgressiveDamageAnalysis,
    ProgressiveDamageResult,
    SmearedCrackDegradation,
)
from .laminate_solver_pipeline import (
    DOF,
    BoundaryConditions,
    ConstraintEliminator,
    DenseLinearSolver,
    FEAMesh,
    GlobalAssembler,
    LinearConstraint,
    LinearSystemSolver,
    LoadCase,
    SingularStiffnessError,
    SolutionResult,
    SparseLinearSolver,
    StructuralGlobalSolver,
)
from .laminate_results import (
    ElementField,
    GaussPointField,
    GlobalTotals,
    NodalField,
    PlyPointField,
    ResultsExtractor,
    extrapolation_matrix,
)
from .laminate_stress_recovery import ExactStressRecoverer, PointStressState

__version__ = "1.0.0"

import types as _types

__all__ = sorted(
    name for name, value in dict(globals()).items()
    if not name.startswith("_") and not isinstance(value, _types.ModuleType)
)
