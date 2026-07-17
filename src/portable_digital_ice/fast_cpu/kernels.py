"""Compiled per-pixel kernels for the optional cpu-fast backend.

Every function here is a line-by-line port of the audited CPU reference: the
same float64 widening, float32 narrowing, and accumulation order, with no
fastmath, no parallel reductions, and no reassociation.  Importing this
module requires numba.
"""

from __future__ import annotations
