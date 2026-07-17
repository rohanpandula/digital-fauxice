"""Compiled execution of the supported streaming profile (optional backend).

The reconstruction math ported here is byte-identical to the CPU reference in
``streaming.py`` and ``reconstruction.py``; only the per-attempted-pixel
scalar chain runs through compiled kernels instead of per-call Python.  Row
analysis (response lookup, auxiliary/score/weighted planes, decision
eligibility), stage-parameter resolution, digesting, and callbacks stay in
Python so behavior outside the compiled chain is unchanged.

Importing this module never requires numba.  Only calling into the compiled
kernels does, and that failure is raised as :class:`CpuFastUnavailable` with a
specific reason instead of silently falling back to a different code path.
"""

from __future__ import annotations


class CpuFastUnavailable(RuntimeError):
    """The compiled CPU backend was requested but cannot run exactly here."""


def _kernels():
    """Import the njit kernel module lazily so package import stays numba-free."""

    try:
        from . import kernels
    except Exception as error:  # pragma: no cover - import environment
        raise CpuFastUnavailable(f"numba is not importable: {error!r}") from error
    return kernels
