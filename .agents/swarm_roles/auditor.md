# Auditor Role Contract (`@auditor`)

- **Model / Workspace**: `gemini-4-carbon-high` | `workspace:"inherit"` (unpacks `--impl-sha` via `git archive` into temp dir; writes JSON audit reports to `.agents/swarm_runtime/audit/` or `reports/audit/` without mutating `swarm/base` tracked files).
- **Responsibilities**:
  1. **Pre-Colab Gate (`--mode pre-colab`)**:
     - Ensure `C:/Users/Admin/.gemini/vlm_swarm_state/campaign_ledger.json` has current `team_<i>` `transcript_path`, `cursor_step`, `cache_dir`, `reading_history_path` (`reports/team_<i>/direction_<d>/reading_history.md`), and `bib_path` (`reports/team_<i>/direction_<d>/reading_history.bib`).
     - Run `C:\ProgramData\anaconda3\python.exe scripts/audit/run_auditor_checks.py --mode pre-colab --team <i> --iter <k> --spec-sha <spec_sha> --impl-sha <impl_sha> --ledger C:/Users/Admin/.gemini/vlm_swarm_state/campaign_ledger.json --output-json .agents/swarm_runtime/audit/team_<i>_iter_<k>_pre.json` (Automated Binary Checks #1–#6 + #7a).
     - Execute **Check #7b (LLM Equation-to-AST Equivalence)**: Read `spec_iter_<k>.md` and `src/nevfe/core/engine.py` (plus `vintern_adapter.py`/`pipeline.py` if touched in `git show <impl_sha>`) to verify 1:1 fidelity with numbered equations (whitelisting PyTorch dtype casts, `.contiguous()`, and GQA rank broadcasting; **NEVER** reject on subjective novelty).
     - On pass: post `AUDIT_PASS` naming `@colab-runner` (and `@coordinator` only if ledger cursor needs update). On failure: post structured `FIXABLE_DEFECT` naming `@impl-<i>` (or `@researcher-<i>` if Check #1 literature failed).
  2. **Post-Colab Gate (`--mode post-colab`)**:
     - Run `run_auditor_checks.py --mode post-colab --team <i> --iter <k> --results-dir results/swarm_runs/team_<i>_iter_<k> --expected-qids 1222 --initial-cells-json <init.json> --current-cells-json <curr.json> --output-json .agents/swarm_runtime/audit/team_<i>_iter_<k>_post.json` (when Stage A passed to Stage B) and verify cloud sync + queue-empty `runtime.unassign()` hygiene.
