"""Provenance-first hybrid repair built around exact Portable Digital ICE."""

from .cache import (
    CachedDiagnostics,
    DiagnosticsCacheBinding,
    DiagnosticsCacheError,
    build_cache_binding,
    load_diagnostics_cache,
    save_diagnostics_cache,
)
from .routing import (
    NoHealthyContextError,
    RoutingPolicy,
    RoutingResult,
    SynthesisBudgetExceeded,
    route_at_floor_mask,
)

__version__ = "0.1.0"

__all__ = [
    "CachedDiagnostics",
    "DiagnosticsCacheBinding",
    "DiagnosticsCacheError",
    "NoHealthyContextError",
    "RoutingPolicy",
    "RoutingResult",
    "SynthesisBudgetExceeded",
    "__version__",
    "build_cache_binding",
    "load_diagnostics_cache",
    "route_at_floor_mask",
    "save_diagnostics_cache",
]
