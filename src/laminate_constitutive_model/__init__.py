"""laminate_constitutive_model - ply and laminate constitutive behaviour of composites.

Public API overview
-------------------
Materials       ElasticProperties3D, StrengthProperties3D, MaterialCard, MaterialLibrary,
                create_default_library, TemperatureDependentMaterial
Laminates       Ply, Laminate; ply stiffness transformations and invariants
Environment     HygrothermalProperties, HygrothermalState, HygrothermalEngine
Kinematics      section stiffness of the through-thickness theories: integrate_abd (ABD),
                CLPTModel, FSDTModel, RefinedZigZagModel, LayerwiseModel
Failure         MaximumStressCriterion, TsaiHillCriterion, TsaiWuCriterion, HashinCriterion,
                DelaminationOnsetCriterion; register_criterion, create_criterion
Verification    PaganoPlate (exact 3D elasticity of a simply supported cross-ply plate)

This package depends on numpy alone. Finite elements, solvers, meshing and blade
models are in ``shell_fem``; beam-section theory is in ``anisotropic_section_model``.
Both depend on this package; it imports neither.
"""

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
from .laminate_temperature import TemperatureDependentMaterial, apply_operating_temperature, laminate_at_temperature
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
from .laminate_hygrothermal import HygrothermalEngine, HygrothermalProperties, HygrothermalState
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

__version__ = "0.2.0"

import types as _types

__all__ = sorted(
    name for name, value in dict(globals()).items()
    if not name.startswith("_") and not isinstance(value, _types.ModuleType)
)
