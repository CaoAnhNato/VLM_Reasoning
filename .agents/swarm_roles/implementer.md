# Implementer Role Contract

- **Roster Name Pattern**: `impl-t<team>`
- **Workspace**: `share` (isolated working copy per team; never share edits concurrently)
- **Responsibilities**:
  - Implement approved spec inside `src/nevfe/` while keeping `src/nevfe/core/` bounded (`<= 2` `.py` files excluding `__init__.py`, `<= 250` lines per file, `AlgorithmConfig <= 3` fields, line 2 `# SPEC_REF: <path>@v1.0 (sha256=<64-hex>)`).
  - Run all local tests ONLY via `python scripts/local/run_guarded.py --idle-timeout-s 1800 -- ...` (`CUDA_VISIBLE_DEVICES=""`, `<= 2` CPU threads, `<= 2.5 GB` RSS) and verify CPU metrics on `data/local_calibration_bank/` via `run_calibration_eval`.
  - Never modify `scripts/benchmark/metrics.py` or `tests/test_metrics_relaxation.py`.
