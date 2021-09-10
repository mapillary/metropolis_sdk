# Copyright (c) Facebook, Inc. and its affiliates.

from .metropolis import Metropolis  # noqa F401

try:
    from ._version import version as __version__  # noqa F401
except ImportError:
    pass
