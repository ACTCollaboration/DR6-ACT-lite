from setuptools import setup
import os
import re

file_dir = os.path.abspath(os.path.dirname(__file__))


def get_version():
    fp = open(os.path.join(file_dir, "act_dr6_cmbonly", "__init__.py")).read()
    version = re.search(r"^__version__ = \"([^\"]*)\"", fp, re.M)

    if version:
        return version.group(1)

    raise RuntimeError("Failed to find version string.")


setup(name="ACT DR6 CMBonly",
      version=get_version(),
      description="Cobaya likelihood for the DR6 foreground-marginalized data",
      author="Hidde T. Jense",
      url="https://github.com/ACTCollaboration/dr6-cmbonly",
      install_requires=["sacc>=0.12.0","cobaya>=3.3"],
      packages = ["act_dr6_cmbonly"],
      package_data={
          "testpaths":"act_dr6_cmbonly"
      })
