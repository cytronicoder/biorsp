import numpy as np

from biorsp.rsp import compute_rsp_profile_from_boolean


def test_rsp_profile_basic():
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    f = np.array([True, False, True, False, True, False, True, False, True, False])
    E_phi, phi_max, E_max = compute_rsp_profile_from_boolean(f, angles, n_bins=5)
    assert E_phi.shape[0] == 5
    assert 0 <= phi_max < 2 * np.pi + 1e-12
    assert np.isclose(E_max, E_phi.max())
