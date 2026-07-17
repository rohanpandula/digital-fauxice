from __future__ import annotations

import struct

import numpy as np

from portable_digital_ice.prepass import reduce_prepass_frame
from portable_digital_ice.producer_parameters import derive_producer_record_schedule
from portable_digital_ice.profile import DEFAULT_PROFILE, ProcessingJob
from portable_digital_ice.rng import LCG24, NIKON_NORMALIZATION
from portable_digital_ice.x3a import (
    NIKON_RESPONSE_LUT_SHA256,
    SharedLookupInputResponse,
)


def test_deterministic_response_and_rng_constants() -> None:
    response = SharedLookupInputResponse.nikon_logarithmic()
    assert response.table.shape == (65_536,)
    assert NIKON_RESPONSE_LUT_SHA256 == (
        "5fd225e25719a544df9475315e3017f9b8027d593ac43e392245de59f49f3fcb"
    )
    assert struct.pack("<f", NIKON_NORMALIZATION) == bytes.fromhex("ffff7f33")
    generator = LCG24.from_nikon_pe_initial_state()
    assert [generator.advance() for _ in range(4)] == [
        0x1791B2,
        0x8223EB,
        0x8B89C0,
        0x2242C1,
    ]


def test_synthetic_prepass_and_producer_are_content_derived(
    supported_job: ProcessingJob,
) -> None:
    response = SharedLookupInputResponse.nikon_logarithmic()
    reduction = reduce_prepass_frame(
        supported_job.acquisition.prepass,
        parameters=DEFAULT_PROFILE.prepass_parameters(),
        frame_index=0,
        evidence_id="synthetic-reduction",
        response=response,
    )
    schedule = derive_producer_record_schedule(
        supported_job.acquisition.main.pixels,
        response.table,
    )

    assert reduction.logical_shape == (8, 8)
    assert reduction.valid_pixels == 64
    assert reduction.fully_valid_tiles == 1
    assert len(reduction.record.payload) == 0x40
    assert schedule.row_count == 8
    assert schedule.mean_schedule.active_width == 8
    assert all(len(payload) == 0x40 for payload in schedule.record_payloads)
    assert np.all(schedule.mean_schedule.accepted_samples == 8)
