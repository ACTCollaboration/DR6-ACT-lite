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
    "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
    "likelihood": {},
    "sampler": {"evaluate": None},
    "debug": True
}


def test_import():
    import act_dr6_cmbonly  # noqa F401


def test_model():
    info["likelihood"] = {
        "act_dr6_cmbonly.ACTDR6CMBonly": None
    }
    model = get_model(info)  # noqa F841
