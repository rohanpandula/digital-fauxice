"""Recovered deterministic random-number primitive used by X3A dither."""

from __future__ import annotations

from dataclasses import dataclass


LCG24_MASK = 0x00FF_FFFF
NIKON_PE_INITIAL_STATE = 0x3045
# ICEDLL.dll VA 0x109ce5f8 stores float32 bits 0x337fffff, the value
# immediately below exact 2**-24.  This is deliberately expressed as an
# exactly representable binary64 hexadecimal literal so the portable runtime
# does not depend on host float32 conversion behavior.
NIKON_NORMALIZATION = float.fromhex("0x1.fffffep-25")


@dataclass
class LCG24:
    """The exact 24-bit state transition recovered at preferred VA 0x1002f920.

    State is per job in the portable implementation.  Nikon's binary stored it
    globally, but a per-job owner preserves a single job's sequence and avoids
    nondeterministic cross-job interference.
    """

    state: int

    def __post_init__(self) -> None:
        if not 0 <= self.state <= LCG24_MASK:
            raise ValueError("LCG24 state must fit in 24 bits")

    @classmethod
    def from_nikon_pe_initial_state(cls) -> "LCG24":
        """Use the proven PE initial value, not an inferred runtime reset."""

        return cls(NIKON_PE_INITIAL_STATE)

    def advance(self) -> int:
        self.state = (125 * self.state + 1) & LCG24_MASK
        return self.state

    def centered_unit(self) -> float:
        """Advance and apply Nikon's biased float32 normalization constant."""

        state = self.advance()
        return (state + 1) * NIKON_NORMALIZATION - 0.5

    def sample(self, center: float, span: float) -> float:
        """Match the recovered ``center + centered_unit * span`` form."""

        return float(center) + self.centered_unit() * float(span)
