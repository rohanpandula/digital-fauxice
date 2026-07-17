"""Host-side derivation of the per-row stage parameters the kernels consume.

The streaming reference resolves three stage calibrations per output row
(source, score, writer).  The CUDA backend precomputes those exact values into
flat arrays so kernels can index them by row.  All quantization happens here
with the same float32 narrowing the reference applies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..reconstruction import ReconstructionParameters
from ..stage_parameters import (
    StageParameterProvider,
    score_stage_for_row,
    source_stage_for_row,
    validate_stage_calibration,
    writer_stage_for_row,
)
from ..x3a import AuxiliaryParameters, ScoreParameters


@dataclass(frozen=True)
class RowParameterTable:
    """Exact per-row calibration mirrors of the streaming stage lookups."""

    aux_alpha: npt.NDArray[np.float32]
    aux_alpha_is_one: npt.NDArray[np.uint8]
    aux_alpha_one_replacement: npt.NDArray[np.float32]
    aux_offset: npt.NDArray[np.float32]
    score_base_primary: npt.NDArray[np.float32]
    writer_coarse_reference: npt.NDArray[np.float32]
    writer_floor_enabled: npt.NDArray[np.uint8]
    writer_row_gate: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        rows = self.aux_alpha.shape[0]
        for name, values, dtype in (
            ("aux_alpha", self.aux_alpha, np.float32),
            ("aux_alpha_is_one", self.aux_alpha_is_one, np.uint8),
            (
                "aux_alpha_one_replacement",
                self.aux_alpha_one_replacement,
                np.float32,
            ),
            ("aux_offset", self.aux_offset, np.float32),
            ("score_base_primary", self.score_base_primary, np.float32),
            ("writer_coarse_reference", self.writer_coarse_reference, np.float32),
            ("writer_floor_enabled", self.writer_floor_enabled, np.uint8),
            ("writer_row_gate", self.writer_row_gate, np.int32),
        ):
            if values.shape != (rows,) or values.dtype != np.dtype(dtype):
                raise ValueError(f"row parameter table field {name} is malformed")


def derive_row_parameter_table(
    height: int,
    *,
    auxiliary_parameters: AuxiliaryParameters,
    score_parameters: ScoreParameters,
    reconstruction_parameters: ReconstructionParameters,
    stage_parameter_provider: StageParameterProvider | None,
) -> RowParameterTable:
    """Resolve the same per-row values streaming resolves, once, up front."""

    aux_alpha = np.empty(height, dtype=np.float32)
    aux_is_one = np.zeros(height, dtype=np.uint8)
    aux_one_replacement = np.zeros(height, dtype=np.float32)
    aux_offset = np.empty(height, dtype=np.float32)
    score_base = np.empty(height, dtype=np.float32)
    writer_reference = np.empty(height, dtype=np.float32)
    writer_floor = np.empty(height, dtype=np.uint8)
    writer_gate = np.empty(height, dtype=np.int32)

    one = np.float32(1.0)
    for row in range(height):
        if stage_parameter_provider is None:
            alpha = np.float32(auxiliary_parameters.alpha)
            offset = np.float32(auxiliary_parameters.calibration_offset)
            replacement = auxiliary_parameters.alpha_one_replacement
            base_primary = np.float32(score_parameters.base_primary)
            coarse_reference = np.float32(
                reconstruction_parameters.coarse_reference
            )
            secondary = reconstruction_parameters.driver_gate_secondary
            row_gate = reconstruction_parameters.row_reconstruction_gate
        else:
            aux_calibration = stage_parameter_provider(source_stage_for_row(row))
            validate_stage_calibration(
                aux_calibration, expected_stage_hit=source_stage_for_row(row)
            )
            score_calibration = stage_parameter_provider(score_stage_for_row(row))
            validate_stage_calibration(
                score_calibration, expected_stage_hit=score_stage_for_row(row)
            )
            writer_calibration = stage_parameter_provider(
                writer_stage_for_row(row)
            )
            validate_stage_calibration(
                writer_calibration, expected_stage_hit=writer_stage_for_row(row)
            )
            alpha = np.float32(aux_calibration.auxiliary_alpha)
            offset = np.float32(aux_calibration.calibration_offset)
            replacement = aux_calibration.base_primary
            base_primary = np.float32(score_calibration.base_primary)
            coarse_reference = np.float32(writer_calibration.base_primary)
            secondary = writer_calibration.writer_gate_secondary
            row_gate = writer_calibration.row_reconstruction_gate

        aux_alpha[row] = alpha
        if alpha == one:
            if replacement is None:
                raise ValueError(
                    "alpha==1 requires the runtime-configured auxiliary replacement"
                )
            aux_is_one[row] = 1
            aux_one_replacement[row] = np.float32(replacement)
        aux_offset[row] = offset
        score_base[row] = base_primary
        writer_reference[row] = coarse_reference
        writer_floor[row] = int(
            reconstruction_parameters.driver_gate_primary or secondary
        )
        writer_gate[row] = row_gate

    return RowParameterTable(
        aux_alpha=aux_alpha,
        aux_alpha_is_one=aux_is_one,
        aux_alpha_one_replacement=aux_one_replacement,
        aux_offset=aux_offset,
        score_base_primary=score_base,
        writer_coarse_reference=writer_reference,
        writer_floor_enabled=writer_floor,
        writer_row_gate=writer_gate,
    )
