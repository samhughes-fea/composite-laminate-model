"""Compatibility facade: kinematic theories and shell elements.

The implementation lives in ``laminate_kinematics`` (theories and section matrices)
and ``laminate_elements`` (the ``ShellElement`` family). This module re-exports both
so code written against the first releases keeps working;
``CompositePlateElement`` is the Quad4 element.
"""

from .laminate_elements import (  # noqa: F401
    DEFAULT_ELEMENTS,
    CompositePlateElement,
    ElementRegistry,
    Quad4ShellElement,
    Quad8ShellElement,
    ShellElement,
    Tri3ShellElement,
    shape_functions,
)
from .laminate_kinematics import (  # noqa: F401
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
    generalized_section,
    harmonic_shear_stiffness,
    integrate_abd,
    integrate_shear,
)
