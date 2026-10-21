"""laminate_modelling_package - generalised composite laminate mechanics suite.

Public API overview
-------------------
Materials       ElasticProperties3D, StrengthProperties3D, MaterialCard, MaterialLibrary,
                TemperatureDependentMaterial
Geometry        Ply, Laminate
Environment     HygrothermalProperties, HygrothermalState, HygrothermalEngine
Kinematics      CLPTModel, FSDTModel, RefinedZigZagModel, LayerwiseModel (full or condensed)
Elements        ShellElement: Quad4ShellElement (alias CompositePlateElement), Tri3ShellElement,
                Quad8ShellElement, CurvedQuad8ShellElement, Tri6ShellElement; ElementRegistry;
                layered solids LayeredHex8Element, LayeredHex20Element (solid_block,
                couple_shell_to_solid)
Lay-up          Layup, PlyDefinition, PlyDrop, ReferenceDirection, AngleField, TabulatedDrape
Orientation     FixedDirection, CylindricalDirection, FieldDirection
Solution        FEAMesh, BoundaryConditions, LoadCase, LoadCombination, StructuralGlobalSolver
Meshing         RectangularPlateBuilder, CylindricalPanelBuilder, GmshPlateBuilder,
                GmshMeshImporter (optional gmsh dependency)
Results         ResultsExtractor (Gauss point, element, node, ply, solid and global levels);
                write_vtu, write_pvd, VTKExporter (ParaView output, per-ply cells)
Post-processing ExactStressRecoverer, EquilibriumStressRecoverer (sigma_z and interlaminar
                shear from 3D equilibrium), failure criteria (register_criterion,
                DelaminationOnsetCriterion), SmearedCrackDegradation, ProgressiveDamageAnalysis
Kernels         NativePythonKernelAdapter, CalculiXKernelAdapter, NastranKernelAdapter
Verification    StressDistributionVerifier, PaganoPlate (exact 3D cross-ply solution),
                ModelChecker (pre-solve checks)
"""

from .laminate_benchmark_stress import DistributionErrorStatistics, StressDistributionVerifier, ThicknessProfile
from .laminate_failure_criteria import (
    FailureCriterion,
    FailureMode,
    FailureResult,
    DelaminationOnsetCriterion,
    HashinCriterion,
    MaximumStressCriterion,
    TsaiHillCriterion,
    TsaiWuCriterion,
    available_criteria,
    create_criterion,
    register_criterion,
)
from .laminate_elasticity_3d import PaganoPlate, PaganoProfile
from .laminate_equilibrium import EquilibriumStressRecoverer, InterfaceCheck
from .laminate_model_check import ModelChecker, ModelIssue, ModelReport
from .laminate_temperature import TemperatureDependentMaterial, apply_operating_temperature, laminate_at_temperature
from .laminate_elements import (
    DEFAULT_ELEMENTS,
    CompositePlateElement,
    ElementRegistry,
    Quad4ShellElement,
    Quad8ShellElement,
    ShellElement,
    Tri3ShellElement,
    shape_functions,
)
from .laminate_kinematics import (
    CLPTModel,
    FSDTModel,
    GeneralizedSection,
    KinematicModel,
    LayerwiseModel,
    LayerwiseThicknessMatrices,
    RefinedZigZagModel,
    SectionModel,
    SectionStiffness,
    ThroughThicknessBasis,
    ThroughThicknessRecovery,
    integrate_abd,
)
from .laminate_mesh_builders import CylindricalPanelBuilder, NamedMesh, RectangularPlateBuilder
from .laminate_orientation import CylindricalDirection, FieldDirection, FixedDirection
from .laminate_gmsh import (
    CircularHole,
    GmshMesh,
    GmshMeshImporter,
    GmshPlateBuilder,
    GmshSession,
    GmshUnavailableError,
    UnsupportedMeshError,
)
from .laminate_hygrothermal import HygrothermalEngine, HygrothermalProperties, HygrothermalState
from .kernels import (
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
    BodyLoad,
    BoundaryConditions,
    EnrichedDOF,
    LineLoad,
    LoadCombination,
    combine_solutions,
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
    SolidPointField,
    extrapolation_matrix,
)
from .laminate_stress_recovery import ExactStressRecoverer, PointStressState, StressRecoverer
from .laminate_curved_shell import CurvedQuad8ShellElement, CurvedShellElement, Tri6ShellElement
from .laminate_layup import AngleField, Layup, PlyDefinition, PlyDrop, ReferenceDirection, TabulatedDrape
from .laminate_solid import (
    DEFAULT_SOLIDS,
    LayeredHex8Element,
    LayeredHex20Element,
    LayeredSolidElement,
    SolidPoint,
    SolidRegistry,
    couple_shell_to_solid,
    solid_block,
)
from .laminate_vtk import VTKExporter, write_pvd, write_vtu

__version__ = "1.0.0"

import types as _types

__all__ = sorted(
    name for name, value in dict(globals()).items()
    if not name.startswith("_") and not isinstance(value, _types.ModuleType)
)
