from __future__ import annotations


class VibeGroupingError(RuntimeError):
    pass


class VibeModelNotFoundError(VibeGroupingError):
    pass


class VibeModelLoadError(VibeGroupingError):
    pass
