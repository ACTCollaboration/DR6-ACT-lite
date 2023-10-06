#!/usr/bin/env python

import os
import sys

from setuptools import setup

TEST_HELP = """
Tests are run via
  tox -e test
I am not very experienced with TOX, so for more information, I recommend
looking at:
  https://github.com/simonsobs/SOLikeT#running-tests
I gracefully stole all this from there.
    - Hidde.
"""

# If for whatever reason we want to update this, here's a version template.
VERSION_TEMPLATE = """
version = "0.1.0"
""".lstrip()

setup(use_scm_version={'write_to': os.path.join('.', 'version.py'),
                       'write_to_template': VERSION_TEMPLATE})
