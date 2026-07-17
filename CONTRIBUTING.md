# Contributing

Fixes, tests, documentation, performance work, and support for new hardware
profiles are welcome. The project also has a strict provenance boundary.

## Keep private material out of the repository

Do not submit:

- Nikon executables, DLLs, firmware, profiles, or installer files;
- decompiler output, disassembly listings, memory dumps, or captured function
  buffers;
- personal scans, scanner captures, comparison oracles, or private paths; or
- tables copied from proprietary files.

Describe behavior through independently written code, public test vectors, and
reproducible observations. If an issue depends on private evidence, describe
the failure without attaching that evidence.

## Development setup

```sh
python -m pip install -e '.[dev]'
ruff check src tests
pytest
python -m build
```

Add a focused test for every behavior change. Unsupported scanner, mode,
resolution, and metric combinations must fail closed.

## Exactness claims

Visual similarity is useful during development, but it is not an exactness
result. A new parity claim needs complete persisted output, source and input
hashes, zero mismatched valid samples, edge coverage, RNG accounting where
applicable, and an independent verifier that does not import the implementation
under test.

## Pull requests

Explain the supported boundary, why the change is safe, and which checks you
ran. Performance changes should report both speed and byte parity. A faster
path that differs from the reference may be useful as an experimental mode,
but it must not replace or silently present itself as the exact path.
