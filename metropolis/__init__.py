# Copyright (c) Facebook, Inc. and its affiliates.

from importlib.metadata import PackageNotFoundError, version

from .metropolis import Metropolis  # noqa F401

try:
    __version__: str = version("metropolis_sdk")
except PackageNotFoundError:
    pass
