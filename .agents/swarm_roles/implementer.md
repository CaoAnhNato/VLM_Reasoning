# Implementer Role Contract (`@impl-1`, `@impl-2`, `@impl-3`)

- **Model / Workspace**: `gemini-4-hydrogen-high` | `workspace:"share"` on branch `swarm/team-<i>` (`i in {1, 2, 3}`).
- **Strict Rules**:
  1. Fetch/checkout `[spec_commit_sha]` on `swarm/team-<i>`. **NEVER** modify `docs/specs/`, `reports/`, `scripts/benchmark/metrics.py`, `tests/test_metrics_relaxation.py`, `main`, `iter23_staging`, or `swarm/base`.
  2. Put `# SPEC_REF: docs/specs/team_<i>/spec_iter_<k>.md@v<ver> (sha256=<SPEC_SHA256>)` on **line 2** of `src/nevfe/core/engine.py`. Keep `src/nevfe/core/` strictly to `__init__.py` + `engine.py` (`<= 250` non-docstring AST statement lines, `AlgorithmConfig <= 3` tunable fields, zero forbidden literals/heuristics/backbone constants).
  3. **NO SILENT FORMULA PATCHING**: If you hit any mathematical ambiguity, `7:1` GQA shape/broadcast mismatch, or `NaN`/`Inf` boundary gap not specified in `spec_iter_<k>.md`, **HALT** and post a `SPEC_DEFECT_REPORT` naming `@researcher-<i>` (wait for `v1.1..v1.3` + updated `SPEC_SHA256`).
  4. Run **ALL** local Python/pytest/calibration commands exclusively via `C:\ProgramData\anaconda3\python.exe scripts/local/run_guarded.py --idle-timeout-s 1800 -- C:\ProgramData\anaconda3\python.exe -m pytest ...` (`CUDA_VISIBLE_DEVICES=""`, `<= 2` CPU threads, `<= 2.5 GB` RSS; never load `1B` weights locally).
  5. Once local unit tests (`test_metrics_relaxation.py`, `test_nevfe_engine.py`, plus new operator/adapter tests) and open-loop calibration finiteness pass, commit `[impl_commit_sha]` onto `swarm/team-<i>` and mention `@auditor` with `{team, iter, spec_sha, impl_sha, ablation_masks}`.
