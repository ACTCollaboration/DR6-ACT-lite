"""
This test checks that the differentiable DR6
likelihood is working as intended.
"""
import pytest  # noqa F401
import numpy as np  # noqa F401


def is_jax_installed():
    try:
        import jax  # noqa F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not is_jax_installed, reason="JAX is not installed.")
def test_import():
    from act_dr6_cmbonly import ACTDR6jax  # noqa F401


@pytest.mark.skipif(not is_jax_installed, reason="JAX is not installed.")
def test_jaxlike():
    import act_dr6_cmbonly
    like = act_dr6_cmbonly.ACTDR6jax()  # noqa F401


if __name__ == "__main__":
    test_import()
    test_jaxlike()
