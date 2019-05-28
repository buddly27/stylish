# :coding: utf-8

import os
import re

from setuptools import setup, find_packages


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCE_PATH = os.path.join(ROOT_PATH, "resource")
SOURCE_PATH = os.path.join(ROOT_PATH, "source")
README_PATH = os.path.join(ROOT_PATH, "README.rst")

PACKAGE_NAME = "stylish"

# Read version from source.
with open(
    os.path.join(SOURCE_PATH, PACKAGE_NAME, "_version.py")
) as _version_file:
    VERSION = re.match(
        r".*__version__ = \"(.*?)\"", _version_file.read(), re.DOTALL
    ).group(1)


# Compute dependencies.
INSTALL_REQUIRES = [
    "click >= 7, < 8",
    "colorama >= 0.3.9, < 1",
    "imageio >= 2.3.0, < 3",
    "numpy >= 1.11.2, < 2",
    "pystache >= 0.5.4, < 1",
    "requests >= 2, < 3",
    "scipy >= 0.1.0, < 2",
    "sawmill >= 0.2.1, < 1",
    "scikit-image >= 0.15.0, < 1",
    "tensorflow >= 1, < 2"
]
DOC_REQUIRES = [
    "changelog >= 0.4, < 1",
    "sphinx >= 1.6, < 2",
    "sphinx-click>=1.2.0",
    "sphinx_rtd_theme >= 0.1.6, < 1"
]
TEST_REQUIRES = [
    "pytest-runner >= 2.7, < 3",
    "pytest >= 4.4.0, < 5",
    "pytest-mock >= 1.1, < 2",
    "pytest-xdist >= 1.1, < 2",
    "pytest-cov >= 2, < 3"
]

setup(
    name="stylish",
    version=VERSION,
    description="Style transfer using deep neural network.",
    long_description=open(README_PATH).read(),
    url="http://github.com/buddly27/stylish",
    keywords=["tensorflow", "style", "transfer", "CNN"],
    author="Jeremy Retailleau",
    packages=find_packages(SOURCE_PATH),
    package_dir={
        "": "source"
    },
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    tests_require=TEST_REQUIRES,
    extras_require={
        "doc": DOC_REQUIRES,
        "test": TEST_REQUIRES,
        "dev": DOC_REQUIRES + TEST_REQUIRES
    },
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "stylish = stylish.__main__:main"
        ]
    },
)
