"""
This test checks that the differentiable DR6
likelihood is working as intended.
"""
import pytest  # noqa F401
import sys

try:
    import jax  # noqa F401
    import jax.numpy as np
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
except ImportError:
    pass


@pytest.mark.skipif("jax" not in sys.modules, reason="JAX is not installed.")
def test_import_jaxlike():
    from act_dr6_cmbonly import ACTDR6jax  # noqa F401


@pytest.mark.skipif("jax" not in sys.modules, reason="JAX is not installed.")
def test_jaxlike():
    import act_dr6_cmbonly
    like = act_dr6_cmbonly.ACTDR6jax()  # noqa F401


@pytest.mark.skipif("jax" not in sys.modules, reason="JAX is not installed.")
def test_jax_load_data():
    import act_dr6_cmbonly
    like = act_dr6_cmbonly.ACTDR6jax()
    like.load_data()


@pytest.mark.skipif("cosmopower_jax" not in sys.modules,
                    reason="Cosmopower-JAX is not installed.")
def test_jax_loglike():
    import act_dr6_cmbonly
    T_CMB = 2.7255e6
    cosmo_params = np.array([0.025, 0.12, 0.68, 0.054, 0.97, 3.05])

    emu_tt = CPJ(probe='cmb_tt')
    emu_te = CPJ(probe='cmb_te')
    emu_ee = CPJ(probe='cmb_ee')
    ellfac = emu_tt.modes * (emu_tt.modes + 1.0) / (2.0 * np.pi)
    cl_tt = (T_CMB) ** 2.0 * emu_tt.predict(cosmo_params) * ellfac
    cl_te = (T_CMB) ** 2.0 * emu_te.predict(cosmo_params) * ellfac
    cl_ee = (T_CMB) ** 2.0 * emu_ee.predict(cosmo_params) * ellfac

    cell = np.stack([cl_tt, cl_te, cl_ee], axis=1)

    like = act_dr6_cmbonly.ACTDR6jax()
    like.load_data()

    logp = like.logp(cell)

    assert np.isclose(logp, -1236.189)


if __name__ == "__main__":
    test_import_jaxlike()
    test_jaxlike()
    test_jax_loglike()
