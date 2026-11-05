"""Array exposures (max stress, Tsai-Wu, Hashin, Puck) and the Puck criterion."""

import numpy as np
import pytest

from laminate_constitutive_model import (
    HashinCriterion,
    MaximumStressCriterion,
    PuckCriterion,
    PuckParameters,
    StrengthProperties3D,
    TsaiWuCriterion,
    create_criterion,
    hashin_exposure,
    max_stress_exposure,
    ply_exposures,
    puck_exposure,
    tsai_wu_exposure,
)

# S23 is Puck's R_TT^A = Yc / (2 (1 + p_TT)) = 80 MPa, as 1d-fem defaults it.
STRENGTH = StrengthProperties3D(Xt=1500e6, Xc=1200e6, Yt=50e6, Yc=200e6, S12=70e6, S23=80e6)

# Eight stress states and the exposures 1d-fem's ply_failure module gave for them
# (onedfem.pre_processing.section_integration.ply_failure at 1d-fem main, 2026-11-05).
STATES = np.array([
    [984122.7, 17924732.3, -8224135.7, -26717755.2, -13640123.6, -49582327.7],
    [48114882.1, 80412914.7, -14766195.6, -18614247.0, 14695261.5, 17844350.4],
    [84331399.2, -55828082.7, -877554.7, 20859095.8, -40326436.4, -22880788.1],
    [-1520978191.8, -77372264.4, -55252051.1, -7052733.9, -38023394.4, 13563217.9],
    [125400869.3, -11215856.7, -75502791.3, -16160786.9, -1455028.4, 5665449.3],
    [-1224108612.4, -28665196.6, -29355572.3, -24265117.2, 31826958.7, -40376733.8],
    [-26017364.0, 53063392.0, -17508013.0, -3351058.5, 3313924.3, 3189088.7],
    [-980044661.1, 4568413.8, 40764702.7, -46414340.3, 25781480.6, 5967701.3],
])
ONEDFEM = {
    "max_stress": [0.7083189671428571, 1.608258294, 0.5760919485714285, 1.2674818264999999,
                   0.37751395649999997, 1.0200905103333333, 1.06126784, 0.8167038842500001],
    "tsai_wu": [0.9153280677033443, 1.5684249491130366, 0.5875132875170147, 0.2947332098757811,
                0.37465730484718707, 0.7297296325710227, 0.9693637946224051, 1.7146208743601463],
    "hashin": [0.8437400671981555, 1.4396254137167972, 0.7146253524394122, 1.2674818264999999,
               0.3514436388144457, 1.0200905103333333, 0.8104980420859971, 1.1280355069889592],
    "puck": [0.9372127859787608, 1.695618003134571, 0.703819523637882, 1.2674818264999999,
             0.3463964436902748, 1.0200905103333333, 1.0655632359510905, 1.4787894340311953],
    "puck_angle": [-15.0772024974511, -9.894926906512278, 50.68827730813821, -69.95237005956764,
                   -52.54763317976568, -36.677245861252175, -2.638928698263857, -56.94741397542464],
}
ONEDFEM_MODES = {
    "max_stress_mode": ["t12", "s22t", "t13", "s11c", "s33c", "s11c", "s22t", "s11c"],
    "hashin_mode": ["MT", "MT", "MC", "FC", "MC", "FC", "MT", "MT"],
    "puck_mode": ["IFF-A", "IFF-A", "IFF-C", "FFC", "IFF-C", "FFC", "IFF-A", "IFF-A"],
}


def test_matches_onedfem_reference():
    out = ply_exposures(STATES, STRENGTH, tsai_wu_interaction="tsai-wu")
    for key, expected in ONEDFEM.items():
        assert np.allclose(out[key], expected, rtol=1e-12, atol=0.0), key
    for key, expected in ONEDFEM_MODES.items():
        assert list(out[key]) == expected, key
    assert np.allclose(out["exposure"], np.max([out[k] for k in ("max_stress", "tsai_wu", "hashin", "puck")], axis=0))
    assert np.allclose(out["margin"], 1.0 / out["exposure"] - 1.0)


def test_arrays_keep_their_shape():
    grid = STATES.reshape(2, 4, 6)
    f, mode, angle = puck_exposure(grid, STRENGTH)
    assert f.shape == mode.shape == angle.shape == (2, 4)
    assert np.allclose(f.ravel(), puck_exposure(STATES, STRENGTH)[0])
    with pytest.raises(ValueError):
        puck_exposure(STATES[:, :5], STRENGTH)


@pytest.mark.parametrize("interaction", ["hoffman", "tsai-wu", 0.3])
def test_tsai_wu_is_the_inverse_strength_ratio(interaction):
    criterion = TsaiWuCriterion(interaction)
    f = tsai_wu_exposure(STATES, STRENGTH, interaction)
    for s, fk in zip(STATES, f):
        assert fk == pytest.approx(1.0 / criterion.evaluate(s, STRENGTH).strength_ratio, rel=1e-12)


def test_max_stress_matches_the_criterion_index():
    f, _ = max_stress_exposure(STATES, STRENGTH)
    for s, fk in zip(STATES, f):
        assert fk == pytest.approx(MaximumStressCriterion().evaluate(s, STRENGTH).index, rel=1e-12)


def test_hashin_tension_modes_match_the_3d_criterion():
    """Tension modes are homogeneous quadratics, so the exposure is the square root of the index."""
    f, mode = hashin_exposure(STATES, STRENGTH)
    names = {"FT": "fiber_tension", "MT": "matrix_tension"}
    checked = 0
    for s, fk, mk in zip(STATES, f, mode):
        if mk in names:
            index = HashinCriterion("3d").evaluate(s, STRENGTH).mode_indices[names[mk]]
            assert fk == pytest.approx(np.sqrt(index), rel=1e-12)
            checked += 1
    assert checked >= 3


def test_exposures_scale_linearly_with_load():
    out1 = ply_exposures(STATES, STRENGTH)
    out3 = ply_exposures(3.0 * STATES, STRENGTH)
    for key in ("max_stress", "tsai_wu", "hashin", "puck"):
        assert np.allclose(out3[key], 3.0 * out1[key], rtol=1e-9), key


def test_puck_uniaxial_strengths():
    zero = np.zeros(6)
    for k, value, expected_mode in ((0, STRENGTH.Xt, "FFT"), (0, -STRENGTH.Xc, "FFC"), (1, STRENGTH.Yt, "IFF-A")):
        s = zero.copy()
        s[k] = value
        f, mode, _ = puck_exposure(s, STRENGTH)
        assert f == pytest.approx(1.0, rel=1e-12)
        assert mode == expected_mode
    s = zero.copy()
    s[5] = STRENGTH.S12
    assert puck_exposure(s, STRENGTH)[0] == pytest.approx(1.0, rel=1e-12)


def test_puck_plane_stress_mode_b_and_c():
    """Puck's closed forms in plane stress: mode B for small compression, mode C beyond."""
    p = PuckParameters()
    R_TT = p.R_TT(STRENGTH)
    S, Yc = STRENGTH.S12, STRENGTH.Yc
    # mode B: |s2 / t21| <= R_TT / |t21c|, f = (sqrt(t21^2 + (p- s2)^2) + p- s2) / S
    s2, t12 = -20e6, 60e6
    expected = (np.sqrt(t12**2 + (p.p_TL_c * s2) ** 2) + p.p_TL_c * s2) / S
    f, mode, angle = puck_exposure(np.array([0, s2, 0, 0, 0, t12]), STRENGTH)
    assert mode == "IFF-B" and abs(angle) < 0.5
    assert f == pytest.approx(expected, rel=1e-6)
    # mode C under pure transverse compression: f = -s2 / Yc on a plane near 50 degrees
    f, mode, angle = puck_exposure(np.array([0, -Yc, 0, 0, 0, 0]), STRENGTH)
    assert mode == "IFF-C"
    assert f == pytest.approx(1.0, rel=1e-3)
    assert abs(angle) == pytest.approx(np.degrees(np.arccos(np.sqrt(R_TT / Yc))), abs=0.5)


def test_puck_criterion_is_registered():
    criterion = create_criterion("puck")
    assert isinstance(criterion, PuckCriterion)
    result = criterion.evaluate(STATES[3], STRENGTH)
    assert result.index == pytest.approx(ONEDFEM["puck"][3], rel=1e-12)
    assert result.strength_ratio == pytest.approx(1.0 / result.index)
    assert result.mode.value == "fiber_compression"
    assert result.failed
    plane = criterion.evaluate([0.0, STRENGTH.Yt * 0.5, 0.0], STRENGTH)  # plane-stress input
    assert plane.index == pytest.approx(0.5, rel=1e-12)
    assert plane.mode.value == "matrix_tension"
    assert plane.active_modes() == set()


def test_puck_parameters_are_checked():
    with pytest.raises(ValueError):
        PuckParameters(p_TT=1.0)
