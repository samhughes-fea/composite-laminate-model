import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import simply_supported
from laminate_modelling_package import (
    DOF,
    AnalysisModel,
    CalculiXDeckWriter,
    CalculiXKernelAdapter,
    CalculiXResultReader,
    FEAMesh,
    FSDTModel,
    HygrothermalState,
    KernelStatus,
    Laminate,
    LoadCase,
    NastranDeckWriter,
    NastranF06Reader,
    NastranFieldFormatter,
    NastranKernelAdapter,
    NativePythonKernelAdapter,
    ProcessRunner,
    StressDistributionVerifier,
    UnsupportedFeatureError,
    create_kernel,
)

A = 0.2


@pytest.fixture
def model(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 45, -45, 90], 0.5e-3)
    mesh = FEAMesh.rectangular_plate(A, A, 6, 6, lam, FSDTModel())
    bcs = simply_supported(mesh, A, A, hard=False)
    loads = LoadCase().add_pressure(1.0e3).set_hygrothermal(HygrothermalState(-30.0, 0.001))
    loads.add_nodal_load(mesh.closest_node((A / 2, A / 2)), DOF.UZ, 5.0)
    return AnalysisModel(mesh, bcs, loads, "unit_model", monitor_elements=(15, 22))


# ----------------------------------------------------------------- Nastran
@pytest.mark.parametrize("value", [1.81e11, 0.28, -100.0, 2.25e-05, 1e-3, 123456789.0, -1.2345678e-7, 7.17e9, 3.0, -0.5])
def test_nastran_real_fields_fit_and_round_trip(value):
    text = NastranFieldFormatter.real(value)
    assert len(text) <= 8
    assert NastranFieldFormatter.parse_real(text) == pytest.approx(value, rel=5e-4)
    assert "." in text


def test_nastran_field_types():
    assert NastranFieldFormatter.field(12) == "12      "
    assert NastranFieldFormatter.field(None) == " " * 8
    assert NastranFieldFormatter.real(0.0) == "0."
    with pytest.raises(ValueError):
        NastranFieldFormatter.field("TOO-LONG-FIELD")
    lines = NastranFieldFormatter.card("SPC1", [1, 123] + list(range(1, 12)))
    assert len(lines) == 2 and lines[1].startswith("+       ")


def test_nastran_deck_structure(model, tmp_path):
    deck = NastranDeckWriter().write(model, tmp_path, "job")
    text = deck.path.read_text().splitlines()
    bulk = text[text.index("BEGIN BULK") + 1 :]
    assert all(len(line) <= 72 for line in bulk if not line.startswith("$"))
    assert sum(line.startswith("GRID") for line in bulk) == model.mesh.n_nodes
    assert sum(line.startswith("CQUAD4") for line in bulk) == model.mesh.n_elements
    assert sum(line.startswith("PCOMP") for line in bulk) == 1
    assert sum(line.startswith("MAT8") for line in bulk) == 1
    assert sum(line.startswith("PLOAD2") for line in bulk) == model.mesh.n_elements
    assert any(line.startswith("TEMPD") for line in bulk)
    assert "TEMPERATURE(LOAD) = 3" in "\n".join(text)
    pcomp = bulk.index(next(line for line in bulk if line.startswith("PCOMP")))
    ply_lines = bulk[pcomp + 1 : pcomp + 3]
    angles = [NastranFieldFormatter.parse_real(line[8 * k : 8 * k + 8]) for line in ply_lines for k in (3, 7)]
    assert angles == [0.0, 45.0, -45.0, 90.0]
    cquad = next(line for line in bulk if line.startswith("CQUAD4"))
    assert cquad[56:64].strip() == "0"  # MCID 0: basic x axis projected onto each element


def test_nastran_equivalent_moisture_expansion(model, cfrp, tmp_path):
    text = NastranDeckWriter().write(model, tmp_path, "job").path.read_text().splitlines()
    mat8 = text.index(next(line for line in text if line.startswith("MAT8")))
    a1 = NastranFieldFormatter.parse_real(text[mat8 + 1][8:16])
    a2 = NastranFieldFormatter.parse_real(text[mat8 + 1][16:24])
    h = cfrp.hygrothermal
    assert a1 == pytest.approx(h.alpha1 + h.beta1 * 0.001 / -30.0, rel=1e-3, abs=1e-12)
    assert a2 == pytest.approx(h.alpha2 + h.beta2 * 0.001 / -30.0, rel=1e-3)


F06_FIXTURE = """1    UNIT_MODEL                                                         OCTOBER  21, 2026  MSC Nastran  2/21/26   PAGE     9
                                                                                                            SUBCASE 1
0                                             D I S P L A C E M E N T   V E C T O R

      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
             1      G      0.0            0.0            0.0            0.0            1.234500E-03   0.0
             2      G      1.000000E-06  -2.000000E-06   3.500000E-04   1.000000E-03  -2.000000E-03   0.0
1    UNIT_MODEL                                                         OCTOBER  21, 2026  MSC Nastran  2/21/26   PAGE    10
0                                             D I S P L A C E M E N T   V E C T O R

      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
             3      G      4.000000E-06   5.000000E-06   6.000000E-04   0.0            0.0            0.0
1    UNIT_MODEL                                                         OCTOBER  21, 2026  MSC Nastran  2/21/26   PAGE    11
                   S T R E S S E S   I N   L A Y E R E D   C O M P O S I T E   E L E M E N T S   ( Q U A D 4 )
   ELEMENT      PLY  STRESSES IN FIBER AND MATRIX DIRECTIONS    INTER-LAMINAR  STRESSES  PRINCIPAL STRESSES (ZERO SHEAR)      MAX
     ID          ID    NORMAL-1     NORMAL-2     SHEAR-12     SHEAR XZ-MAT  SHEAR YZ-MAT  ANGLE    MAJOR        MINOR        SHEAR
        15        1  -4.66E+06    -1.30E+06     2.69E+05      1.0E+03      2.0E+03    4.2   -1.29E+06    -4.68E+06     1.69E+06
        15        2  -2.77E+06    -5.48E+05    -6.90E+05      3.0E+03      4.0E+03  -16.3   -3.49E+05    -2.97E+06     1.31E+06
"""


def test_f06_reader(tmp_path):
    path = tmp_path / "unit.f06"
    path.write_text(F06_FIXTURE)
    disp, plies, fatal = NastranF06Reader().read(path)
    assert sorted(disp) == [1, 2, 3]
    assert disp[1][4] == pytest.approx(1.2345e-3)
    assert disp[3][2] == pytest.approx(6e-4)
    assert np.allclose(plies[15][2], [-2.77e6, -5.48e5, -6.90e5, 4.0e3, 3.0e3])
    assert fatal == []
    profile = NastranF06Reader.thickness_profile(15, plies[15], np.array([-0.75e-3, -0.25e-3, 0.25e-3, 0.75e-3]))
    assert profile.frame == "material" and profile.values.shape == (2, 5)


class FakeNastranRunner(ProcessRunner):
    """Stands in for the nastran executable by writing a canned .f06."""

    def __init__(self, f06_text):
        self.f06_text = f06_text
        self.calls = []

    def run(self, args, cwd, timeout):
        self.calls.append(list(args))
        stem = Path(args[1]).stem
        (Path(cwd) / f"{stem}.f06").write_text(self.f06_text)
        return subprocess.CompletedProcess(args, 0, "", "")


def test_nastran_adapter_flow_with_stub_runner(model, tmp_path, monkeypatch):
    n = model.mesh.n_nodes
    rows = "\n".join(
        f"        {i:6d}      G      1.0E-06        2.0E-06        {i * 1.0e-6:.6E}   0.0            0.0            0.0"
        for i in range(1, n + 1)
    )
    f06 = "0                                             D I S P L A C E M E N T   V E C T O R\n\n" + rows + "\n"
    runner = FakeNastranRunner(f06)
    adapter = NastranKernelAdapter(work_dir=tmp_path, runner=runner)
    monkeypatch.setattr(adapter, "executable_path", lambda: "/opt/fake/nastran")
    result = adapter.run(model)
    assert result.status is KernelStatus.SUCCESS, result.message
    assert runner.calls[0][1] == "unit_model.bdf"
    assert result.max_abs_displacement(2) == pytest.approx(n * 1e-6)


def test_nastran_fatal_is_reported(model, tmp_path, monkeypatch):
    runner = FakeNastranRunner("*** USER FATAL MESSAGE 2101 (GP1):  bad grid\n")
    adapter = NastranKernelAdapter(work_dir=tmp_path, runner=runner)
    monkeypatch.setattr(adapter, "executable_path", lambda: "/opt/fake/nastran")
    result = adapter.run(model)
    assert result.status is KernelStatus.FAILED and "FATAL" in result.message


# ----------------------------------------------------------------- CalculiX
def test_calculix_deck_promotes_to_s8r(model, tmp_path):
    deck = CalculiXDeckWriter().write(model, tmp_path, "job")
    text = deck.path.read_text()
    n, e = 6, 6
    assert len(deck.midside_nodes) == n * (e + 1) * 2
    assert "*ELEMENT, TYPE=S8R, ELSET=EALL" in text
    assert "*SHELL SECTION, ELSET=SEC1, COMPOSITE, OFFSET=0" in text
    assert "1, 0, 0, 0, 1, 0" in text  # flat mesh: material axes are global x, y
    assert text.count("*ORIENTATION") == 4
    assert "*EXPANSION, TYPE=ORTHO, ZERO=0." in text and "*TEMPERATURE" in text
    # Mid-side nodes between two simply supported edge nodes are supported too.
    corner_a, corner_b = 1, 2
    m = deck.midside_nodes[(corner_a, corner_b)]
    assert f"\n{m}, 3, 3, 0" in text
    assert len(deck.layer_heights[15]) == 8


def test_calculix_rejects_gradient_states(model, tmp_path):
    model.load_case.set_hygrothermal(HygrothermalState(-30.0, temperature_gradient=1e3))
    with pytest.raises(UnsupportedFeatureError):
        CalculiXDeckWriter().write(model, tmp_path, "job")


def test_calculix_dat_reader(tmp_path):
    lines = [
        "",
        "                        S T E P       1",
        " displacements (vx,vy,vz) for set NALL and time  0.1000000E+01",
        "",
        "         1  0.000000E+00  0.000000E+00  0.000000E+00",
        "         2 -2.038103E-05  9.440893E-07  4.906632E-03",
        "",
        " stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EMON and time  0.1000000E+01",
        "",
    ]
    for ip in range(1, 17):
        level = (ip - 1) // 4
        lines.append(f"         7  {ip:2d}  {level + 1:.6E}  {10 * (level + 1):.6E}  0.0E+00  {100.0:.6E}  {1.0:.6E}  {2.0:.6E} OR1_shell")
    path = tmp_path / "x.dat"
    path.write_text("\n".join(lines))
    reader = CalculiXResultReader()
    disp, stresses = reader.read(path)
    assert disp[2][2] == pytest.approx(4.906632e-3)
    profile = reader.thickness_profile(7, stresses[7], np.array([-0.3, -0.1, 0.1, 0.3]))
    assert np.allclose(profile.component("s11"), [1, 2, 3, 4])
    assert np.allclose(profile.component("t13"), 1.0) and np.allclose(profile.component("t23"), 2.0)


def test_missing_executables_are_skipped_with_deck(model, tmp_path):
    for adapter in (
        CalculiXKernelAdapter(executable="no-such-ccx-binary", work_dir=tmp_path / "c"),
        NastranKernelAdapter(executable="no-such-nastran-binary", work_dir=tmp_path / "n"),
    ):
        result = adapter.run(model)
        assert result.status is KernelStatus.SKIPPED
        assert result.files["deck"].exists()
        assert result.runtime >= 0.0


def test_native_kernel_and_registry(model):
    result = create_kernel("native").run(model)
    assert result.ok and len(result.displacements) == model.mesh.n_nodes
    assert set(result.stress_profiles[15]) == {"global", "material", "material_midply"}
    assert result.profile(15, "material").values.shape == (8, 5)
    assert isinstance(create_kernel("CalculiX"), CalculiXKernelAdapter)
    with pytest.raises(KeyError):
        create_kernel("abaqus")


def test_kernel_failures_are_captured(model):
    model.monitor_elements = (999,)
    result = NativePythonKernelAdapter().run(model)
    assert result.status is KernelStatus.FAILED and "KeyError" in result.message


@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX (ccx) is not installed")
def test_calculix_agrees_with_native_solver(model, tmp_path):
    native = NativePythonKernelAdapter().run(model)
    ccx = CalculiXKernelAdapter(work_dir=tmp_path).run(model)
    assert ccx.ok, ccx.message
    verifier = StressDistributionVerifier()
    stats = verifier.compare_displacements(native.displacements, ccx.displacements)
    assert stats["UZ"].normalized_max_error < 0.05
    assert native.max_abs_displacement(2) == pytest.approx(ccx.max_abs_displacement(2), rel=0.05)
    profile = verifier.compare_profiles(native.profile(15, "material"), ccx.profile(15, "material"), ("s11", "s22", "t12"))
    for name, stat in profile.items():
        assert stat.normalized_max_error < 0.1, (name, stat)
