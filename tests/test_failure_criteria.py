import numpy as np
import pytest

from composite_laminate_model import (
    FailureMode,
    HashinCriterion,
    MaximumStressCriterion,
    StrengthProperties3D,
    TsaiHillCriterion,
    TsaiWuCriterion,
    create_criterion,
)

S = StrengthProperties3D(Xt=1500e6, Xc=1200e6, Yt=40e6, Yc=200e6, S12=70e6, Zt=45e6, Zc=210e6, S13=75e6, S23=50e6)
ALL = [MaximumStressCriterion(), TsaiHillCriterion(), TsaiWuCriterion(), HashinCriterion("2d"), HashinCriterion("3d")]


def voigt(s1=0.0, s2=0.0, s3=0.0, t23=0.0, t13=0.0, t12=0.0):
    return np.array([s1, s2, s3, t23, t13, t12])


@pytest.mark.parametrize("criterion", ALL, ids=lambda c: c.name)
@pytest.mark.parametrize(
    "stress",
    [voigt(s1=1500e6), voigt(s1=-1200e6), voigt(s2=40e6), voigt(s2=-200e6), voigt(t12=70e6)],
    ids=["Xt", "Xc", "Yt", "Yc", "S12"],
)
def test_uniaxial_strengths_give_unit_index(criterion, stress):
    result = criterion.evaluate(stress, S)
    assert result.index == pytest.approx(1.0, rel=1e-9)
    assert result.strength_ratio == pytest.approx(1.0, rel=1e-9)
    assert result.failed


@pytest.mark.parametrize("criterion", ALL, ids=lambda c: c.name)
def test_strength_ratio_scales_to_failure_surface(criterion):
    stress = voigt(s1=300e6, s2=12e6, t12=-25e6, t13=5e6)
    result = criterion.evaluate(stress, S)
    at_failure = criterion.evaluate(result.strength_ratio * stress, S)
    assert at_failure.index == pytest.approx(1.0, rel=1e-8)
    assert not result.failed


def test_hoffman_interaction_coefficient():
    F, FF = TsaiWuCriterion("hoffman").coefficients(S)
    assert FF[0, 1] == pytest.approx(-0.5 * (1 / (S.Xt * S.Xc) + 1 / (S.Yt * S.Yc) - 1 / (S.Zt * S.Zc)))
    plane = StrengthProperties3D(1500e6, 1200e6, 40e6, 200e6, 70e6)
    _, FF_plane = TsaiWuCriterion("hoffman").coefficients(plane)
    assert FF_plane[0, 1] == pytest.approx(-1.0 / (2 * plane.Xt * plane.Xc))
    _, FF_tw = TsaiWuCriterion("tsai-wu").coefficients(plane)
    assert FF_tw[0, 1] == pytest.approx(-0.5 / np.sqrt(plane.Xt * plane.Xc * plane.Yt * plane.Yc))
    with pytest.raises(ValueError):
        TsaiWuCriterion("bogus")


def test_tsai_hill_reduces_to_plane_stress_form():
    s1, s2, t12 = 500e6, 15e6, 30e6
    expected = (s1 / S.Xt) ** 2 - s1 * s2 / S.Xt**2 + (s2 / S.Yt) ** 2 + (t12 / S.S12) ** 2
    plane = StrengthProperties3D(S.Xt, S.Xc, S.Yt, S.Yc, S.S12)
    assert TsaiHillCriterion().evaluate([s1, s2, t12], plane).index == pytest.approx(expected)


def test_hashin_mode_identification():
    h3 = HashinCriterion("3d")
    assert h3.evaluate(voigt(s1=1600e6), S).mode is FailureMode.FIBER_TENSION
    assert h3.evaluate(voigt(s1=-1300e6), S).mode is FailureMode.FIBER_COMPRESSION
    assert h3.evaluate(voigt(s2=45e6), S).mode is FailureMode.MATRIX_TENSION
    assert h3.evaluate(voigt(s2=-220e6), S).mode is FailureMode.MATRIX_COMPRESSION
    delam = h3.evaluate(voigt(t13=60e6, t23=35e6), S)
    assert delam.mode_indices["delamination"] == pytest.approx((60 / 75) ** 2 + (35 / 50) ** 2)
    assert h3.evaluate(voigt(s3=45e6), S).mode_indices["delamination"] == pytest.approx(1.0)


def test_hashin_2d_matrix_compression_formula():
    s2, t12 = -120e6, 30e6
    expected = (s2 / (2 * S.S23)) ** 2 + ((S.Yc / (2 * S.S23)) ** 2 - 1) * s2 / S.Yc + (t12 / S.S12) ** 2
    result = HashinCriterion("2d").evaluate(voigt(s2=s2, t12=t12), S)
    assert result.mode_indices["matrix_compression"] == pytest.approx(expected)
    # With s1 = 0 the fibre-tension mode (shear-driven) governs here.
    assert result.mode is FailureMode.FIBER_TENSION


def test_max_stress_reports_governing_mode():
    result = MaximumStressCriterion().evaluate(voigt(s1=100e6, s2=30e6, t12=10e6), S)
    assert result.mode is FailureMode.MATRIX_TENSION
    assert result.index == pytest.approx(0.75)


def test_plane_stress_input_and_factory():
    assert create_criterion("tsai_wu").evaluate([0.0, 0.0, 0.0], S).index == 0.0
    assert create_criterion("hashin_2d").name == "Hashin2D"
    with pytest.raises(KeyError):
        create_criterion("nope")
    with pytest.raises(ValueError):
        MaximumStressCriterion().evaluate([1.0, 2.0], S)
