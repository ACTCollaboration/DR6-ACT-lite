"""
This test checks that the ACT DR6 CMB-only likelihood is correctly installed
and working properly.
"""
import os
import pytest  # noqa F401
import numpy as np
from cobaya.model import get_model

info = {
    "params": {
        "ombh2": 0.022,
        "omch2": 0.117,
        "ns": 0.96,
        "As": 2e-9,
        "tau": 0.065,
        "cosmomc_theta": 104.09e-4
    },
    "theory": {"camb": {"extra_args": {"lmax": 9000, "lens_potential_accuracy": 8, "min_l_logl_sampling": 6000}}},
    "likelihood": {},
    "sampler": {"evaluate": None},
    "debug": True
}


def test_import():
    import act_dr6_cmbonly  # noqa F401


def test_model():
    info["likelihood"] = {
        "act_dr6_cmbonly.ACTDR6CMBonly": {
            "input_file" : "act_dr6_cmb_sacc.fits"
        }
    }
    model = get_model(info)  # noqa F841

def test_TTTEEE():
    info["likelihood"] = {
        "act_dr6_cmbonly.ACTDR6CMBonly": {
            "stop_at_error": True,
            "input_file": "act_dr6_cmb_sacc.fits"
        }
    }
    model = get_model(info)
    loglikes = model.loglikes()[0][0]
    assert np.isclose(loglikes, -1356.91), "TT/TE/EE log-posterior does not match."

if __name__ == "__main__":
    test_import()
    test_model()
    test_TTTEEE()
