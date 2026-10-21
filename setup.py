"""Packaging script for laminate_modelling_package.

Editable install for development:
    pip install -e .            # core (NumPy only)
    pip install -e .[sparse]    # with SciPy sparse solver
    pip install -e .[dev]       # with SciPy and pytest
"""

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setup(
    name="laminate_modelling_package",
    version="1.0.0",
    description=(
        "Generalised composite laminate mechanics: CLPT/FSDT/zig-zag/layerwise kinematics, "
        "Quad4 shell FEA, hygrothermal loads, failure criteria, progressive damage and "
        "native/CalculiX/Nastran solver kernels."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/samhughes-fea/composite-laminate-fea",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.9",
    install_requires=["numpy>=1.22"],
    extras_require={
        "sparse": ["scipy>=1.8"],
        "dev": ["scipy>=1.8", "pytest>=7"],
    },
    entry_points={"console_scripts": ["laminate-benchmarks=run_solver_benchmarks:cli"]},
    py_modules=["run_solver_benchmarks"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
)
