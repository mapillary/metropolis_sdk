# Copyright (c) Facebook, Inc. and its affiliates.

from .metropolis import Metropolis  # noqa F401

try:
    # pyre-fixme[21]: Could not find module
    #  `mapillary.research.metropolis.metropolis._version`.
    from ._version import version as __version__  # noqa F401
except ImportError:
    pass
