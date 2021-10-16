# Copyright (c) Facebook, Inc. and its affiliates.

from os import path

import setuptools

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    # Meta-data
    name="metropolis_sdk",
    author="Lorenzo Porzi",
    author_email="porzi@fb.com",
    description="Mapillary Metropolis SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mapillary/metropolis_sdk",
    # Versioning
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "write_to": "metropolis/_version.py",
    },
    # Requirements
    setup_requires=["setuptools_scm"],
    install_requires=[
        "matplotlib",
        "numpy",
        "pillow",
        "pyquaternion",
        "iopath",
        "scikit-image",
        "open3d",
    ],
    python_requires=">=3, <4",
    # Package description
    packages=[
        "metropolis",
        "metropolis.utils",
    ],
    # Scripts
    scripts=["scripts/metropolis_crop_point_clouds"],
)
