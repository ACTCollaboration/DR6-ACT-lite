from setuptools import setup
from act_dr6_cmbonly import __author__, __version__


setup(name="ACT DR6 CMBonly",
      version=__version__,
      description="Cobaya likelihood for the DR6 foreground-marginalized data",
      author=__author__,
      url="https://github.com/ACTCollaboration/dr6-cmbonly",
      install_requires=["sacc>=0.12.0", "cobaya>=3.3"],
      extra_requires={"jax": ["jax"]},
      packages=["act_dr6_cmbonly"],
      package_data={"act_dr6_cmbonly": ["*.yaml", "data/*", "tests/*"]},
      test_suite="act_dr6_cmbonly.tests",
      classifiers=[
          "Programming Language :: Python :: 3"
      ])
