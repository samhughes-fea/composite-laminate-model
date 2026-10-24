import numpy as np
import pytest

from laminate_modelling_package import (
    DuplicateMaterialError,
    ElasticProperties3D,
    MaterialCard,
    MaterialLibrary,
    MaterialNotFoundError,
    StrengthProperties3D,
    rotate_stiffness_3d,
    strain_rotation_3d,
    stress_rotation_3d,
)


def test_compliance_is_symmetric_and_inverse_of_stiffness(cfrp):
    S = cfrp.elastic.compliance_matrix()
    assert np.allclose(S, S.T)
    assert np.allclose(cfrp.elastic.stiffness_matrix() @ S, np.eye(6), atol=1e-10)


def test_reduced_stiffness_matches_closed_form(cfrp):
    e = cfrp.elastic
    Q = e.reduced_stiffness()
    d = 1 - e.v12 * e.v21
    assert Q[0, 0] == pytest.approx(e.E1 / d)
    assert Q[1, 1] == pytest.approx(e.E2 / d)
    assert Q[0, 1] == pytest.approx(e.v12 * e.E2 / d)
    assert Q[2, 2] == pytest.approx(e.G12)


def test_reduced_stiffness_equals_condensed_3d_compliance(cfrp):
    S = cfrp.elastic.compliance_matrix()
    idx = [0, 1, 5]
    assert np.allclose(np.linalg.inv(S[np.ix_(idx, idx)]), cfrp.elastic.reduced_stiffness())


def test_isotropic_stiffness_matches_lame_constants():
    E, nu = 70e9, 0.3
    C = ElasticProperties3D.isotropic(E, nu).stiffness_matrix()
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    assert C[0, 0] == pytest.approx(lam + 2 * mu)
    assert C[0, 1] == pytest.approx(lam)
    assert C[3, 3] == pytest.approx(mu)


def test_transversely_isotropic_shear_modulus():
    e = ElasticProperties3D.transversely_isotropic(150e9, 10e9, 0.3, 5e9, 0.45)
    assert e.G23 == pytest.approx(10e9 / 2.9)
    assert e.E3 == e.E2 and e.G13 == e.G12


@pytest.mark.parametrize("bad", [dict(E1=-1.0), dict(G12=0.0), dict(v12=5.0)])
def test_invalid_elastic_constants_are_rejected(cfrp, bad):
    with pytest.raises(ValueError):
        cfrp.elastic.with_updates(**bad)


def test_elastic_properties_are_frozen(cfrp):
    with pytest.raises(Exception):
        cfrp.elastic.E1 = 1.0


def test_strength_defaults_follow_transverse_isotropy():
    s = StrengthProperties3D(1500e6, 1200e6, 50e6, 200e6, 70e6)
    assert (s.Zt, s.Zc, s.S13, s.S23) == (50e6, 200e6, 70e6, 70e6)
    with pytest.raises(ValueError):
        StrengthProperties3D(1500e6, -1.0, 50e6, 200e6, 70e6)


def test_library_repository_behaviour(cfrp):
    lib = MaterialLibrary()
    lib.register(cfrp)
    assert "T300/5208" in lib and len(lib) == 1
    assert lib.get("T300/5208") is cfrp
    with pytest.raises(DuplicateMaterialError):
        lib.register(cfrp)
    lib.register(cfrp, overwrite=True)
    lib.remove("T300/5208")
    with pytest.raises(MaterialNotFoundError):
        lib.get("T300/5208")
    with pytest.raises(TypeError):
        lib.register("not a card")


def test_default_library_contents(library):
    assert {"T300/5208", "AS4/3501-6", "IM7/8552", "E-glass/epoxy", "Al-6061-T6"} <= set(library.names())
    for card in library:
        assert isinstance(card, MaterialCard)
        assert card.hygrothermal is not None


@pytest.mark.parametrize("theta", [0.0, 25.0, 45.0, 90.0, -60.0])
def test_3d_rotations_are_consistent(cfrp, theta):
    T_s, T_e = stress_rotation_3d(theta), strain_rotation_3d(theta)
    # Energy invariance: sigma . eps is frame independent  <=>  T_s^T T_e^-T ... equivalently T_e^-1 = T_s^T
    assert np.allclose(np.linalg.inv(T_e), T_s.T)
    C = cfrp.elastic.stiffness_matrix()
    assert np.allclose(rotate_stiffness_3d(rotate_stiffness_3d(C, theta), -theta), C, rtol=1e-10, atol=1e-9 * C.max())
