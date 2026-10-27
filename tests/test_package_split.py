"""The three packages and the boundaries between them.

    blade_geometry_model  <-  composite_laminate_model  <-  anisotropic_section_model

- The geometry imports nothing from the suite.
- The laminate model may import the geometry, never the section model.
- The section model may import the geometry, and from the laminate model only
  ``Laminate`` and ``integrate_abd``: walls are laminates, but section theory never
  touches elements, the solver, kernels or ``beam_section_properties`` (shell
  physics and beam-section physics meet only in tests and examples).
- Nothing imports the ``laminate_modelling_package`` shim.

Every import is checked, including ones inside functions.
"""

import ast
import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
GEOMETRY, LAMINATE, SECTION, SHIM = (
    "blade_geometry_model",
    "composite_laminate_model",
    "anisotropic_section_model",
    "laminate_modelling_package",
)
SUITE = {GEOMETRY, LAMINATE, SECTION, SHIM}

# what the section model may take from the laminate model: module -> names
SECTION_FROM_LAMINATE = {
    LAMINATE: {"Laminate", "integrate_abd"},
    f"{LAMINATE}.laminate_mechanics": {"Laminate"},
    f"{LAMINATE}.laminate_kinematics": {"integrate_abd"},
}


def _modules(package):
    root = SRC / package
    for path in sorted(root.rglob("*.py")):
        yield path, ".".join(path.relative_to(SRC).with_suffix("").parts)


def _imports(path, module):
    """(imported module, names or None, line) for every import in a file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    package_parts = module.split(".")
    if path.name != "__init__.py":
        package_parts = package_parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - node.level + 1]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module
            yield target, {a.name for a in node.names}, node.lineno


def _violations(package):
    found = []
    for path, module in _modules(package):
        for target, names, line in _imports(path, module):
            top = target.split(".")[0]
            if top not in SUITE or top == package:
                continue
            where = f"{path.relative_to(SRC)}:{line} imports {target}"
            if top == SHIM:
                found.append(f"{where} (the compatibility shim)")
            elif package == GEOMETRY:
                found.append(where)
            elif package == LAMINATE and top == SECTION:
                found.append(where)
            elif package == SECTION and top == LAMINATE:
                allowed = SECTION_FROM_LAMINATE.get(target)
                if allowed is None or names is None or not names <= allowed:
                    found.append(f"{where} {sorted(names or [])}")
    return found


@pytest.mark.parametrize("package", [GEOMETRY, LAMINATE, SECTION])
def test_import_boundaries(package):
    assert _violations(package) == []


def test_the_checker_catches_a_crossing(tmp_path, monkeypatch):
    """A section module reaching for the shell solver is flagged."""
    pkg = tmp_path / SECTION
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "bad.py").write_text(
        "from composite_laminate_model.laminate_mechanics import Laminate\n"
        "def f():\n"
        "    from composite_laminate_model.laminate_blade import beam_section_properties\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path)
    found = _violations(SECTION)
    assert len(found) == 1 and "laminate_blade" in found[0]


def test_geometry_imports_on_its_own():
    """Importing the geometry loads nothing else from the suite."""
    import subprocess

    code = (
        "import sys; import blade_geometry_model; "
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in {sorted(SUITE - {GEOMETRY})!r}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env={"PYTHONPATH": str(SRC)}
    )
    assert out.stdout.strip() == "[]"


def test_versions_match():
    versions = {p: importlib.import_module(p).__version__ for p in (GEOMETRY, LAMINATE, SECTION, SHIM)}
    assert len(set(versions.values())) == 1, versions


def test_shim_reexports_the_same_objects():
    shim = importlib.import_module(SHIM)
    owners = [importlib.import_module(p) for p in (GEOMETRY, LAMINATE, SECTION)]
    for name in shim.__all__:
        value = getattr(shim, name)
        assert any(getattr(m, name, None) is value for m in owners), name


@pytest.mark.parametrize(
    "old, new",
    [
        ("laminate_gbt", f"{SECTION}.gbt"),
        ("laminate_thin_wall", f"{SECTION}.thin_wall"),
        ("laminate_torsion", f"{SECTION}.torsion"),
        ("laminate_beam", f"{SECTION}.beam"),
        ("laminate_blade", f"{LAMINATE}.laminate_blade"),
        ("laminate_materials_3d", f"{LAMINATE}.laminate_materials_3d"),
        ("kernels.calculix", f"{LAMINATE}.kernels.calculix"),
    ],
)
def test_shim_submodules_are_aliases(old, new):
    assert importlib.import_module(f"{SHIM}.{old}") is importlib.import_module(new)


def test_public_api_split():
    """Each public name has one home, and the shim's list is their union."""
    shim = importlib.import_module(SHIM)
    geometry, laminate, section = (importlib.import_module(p) for p in (GEOMETRY, LAMINATE, SECTION))
    assert set(section.__all__).isdisjoint(laminate.__all__)
    builder_inputs = {"Airfoil", "Station", "SkinRegion", "ShearWeb"}  # re-exported by the laminate model
    assert builder_inputs <= set(geometry.__all__) & set(laminate.__all__)
    for name in builder_inputs:
        assert getattr(laminate, name) is getattr(geometry, name)
    assert set(laminate.__all__).isdisjoint(set(geometry.__all__) - builder_inputs)
    assert set(shim.__all__) == set(laminate.__all__) | set(section.__all__)
