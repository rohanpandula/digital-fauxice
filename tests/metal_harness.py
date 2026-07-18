"""Shared Metal dispatch plumbing for the metal_backend test files.

Wraps the pyobjc launch boilerplate so the tests read like the CUDA tests:
compile a library (the backend source plus test wrappers), then launch one
kernel over flat numpy arrays and read flat numpy arrays back.  Not a test
file; pytest collects only ``test_*.py``.
"""

from __future__ import annotations

import numpy as np


class MetalHarness:
    def __init__(self, source: str) -> None:
        import Metal

        from portable_digital_ice.metal_backend.engine import compile_library

        self._metal = Metal
        self._library = compile_library(source)
        from portable_digital_ice.metal_backend.engine import _device

        _, self._device = _device()
        self._queue = self._device.newCommandQueue()
        self._pipelines: dict[str, object] = {}

    def _pipeline(self, name: str):
        pipeline = self._pipelines.get(name)
        if pipeline is None:
            function = self._library.newFunctionWithName_(name)
            assert function is not None, f"kernel {name} missing"
            pipeline, error = (
                self._device.newComputePipelineStateWithFunction_error_(
                    function, None
                )
            )
            assert error is None, error
            self._pipelines[name] = pipeline
        return pipeline

    def buffer_from(self, array: np.ndarray):
        array = np.ascontiguousarray(array)
        buf = self._device.newBufferWithLength_options_(
            max(array.nbytes, 16), self._metal.MTLResourceStorageModeShared
        )
        view = np.frombuffer(
            buf.contents().as_buffer(max(array.nbytes, 16)), dtype=array.dtype
        )
        view[: array.size] = array.reshape(-1)
        return buf

    def buffer_out(self, dtype, count: int):
        nbytes = max(int(count) * np.dtype(dtype).itemsize, 16)
        buf = self._device.newBufferWithLength_options_(
            nbytes, self._metal.MTLResourceStorageModeShared
        )
        return buf

    def read(self, buf, dtype, count: int) -> np.ndarray:
        nbytes = int(count) * np.dtype(dtype).itemsize
        return np.frombuffer(
            buf.contents().as_buffer(nbytes), dtype=dtype
        ).copy()

    def run(self, name: str, buffers, grid) -> None:
        pipeline = self._pipeline(name)
        command_buffer = self._queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, index)
        width = min(256, int(pipeline.maxTotalThreadsPerThreadgroup()))
        if isinstance(grid, tuple):
            threads = self._metal.MTLSizeMake(grid[0], grid[1], 1)
            group = self._metal.MTLSizeMake(32, max(1, width // 32), 1)
        else:
            threads = self._metal.MTLSizeMake(int(grid), 1, 1)
            group = self._metal.MTLSizeMake(width, 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(threads, group)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        assert command_buffer.status() == 4, command_buffer.error()
