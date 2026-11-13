# Coordinator Role Contract (`@coordinator`)

- **Model / Workspace**: `gemini-4-carbon-high` | `workspace:"inherit"` (read-only repo on `swarm/base`; writes `C:/Users/Admin/.gemini/vlm_swarm_state/campaign_ledger.json`).
- **Roster (11 Agents Total)**:
  - `gemini-4-carbon-high`: `@coordinator`, `@critic`, `@auditor`, `@researcher-1`, `@researcher-2`, `@researcher-3`
  - `gemini-4-hydrogen-high`: `@impl-1`, `@impl-2`, `@impl-3`, `@colab-runner`, `@monitor`
- **Responsibilities**:
  - Maintain `C:/Users/Admin/.gemini/vlm_swarm_state/campaign_ledger.json` (agent UUIDs, worktree paths, `direction_id`, `iter_k` up to 10, `strikes`, `best_commit`, `cursor_step`, `cu_spent_est <= 65.0`).
  - Arbitrate 4-line Hypothesis Registry ties on the `/swarm` board by board sequence order across disjoint stages: Team 1 (Input Optical & Native RGB Sensing), Team 2 (Autoregressive Trajectory & Output Distribution), Team 3 (Intra-Decoder Cross-Modal Routing & Latent Subspace).
  - Handle `@monitor [stall]` alerts: Level-1 (`>= 30m` `delta == 0` with live `child_pid`) -> verify in `transcript.jsonl` and run `taskkill /PID <child_pid> /T /F` without `swarm(action="kill")`; Level-2 (`>= 60m` `delta == 0` with `0` child PIDs) -> `swarm(action="add", name="<role>-2", fork="<role>", workspace="share")` then `swarm(action="kill", name="<role>")` + `git worktree prune`.
  - Adjudicate `v1.3` spec deadlocks (`git checkout <team_best_commit> -- src/nevfe/ tests/`, `k -> k+1`, `+1` strike), enforce 3-strike direction pivots, enforce `<= 65 CU` budget ceiling, and post `[done] <summary>` (no `@`) + confirm `runtime.unassign()` upon reaching `> 70.0%` Token F1 & `> 70.0%` Substring Match or finishing Iteration 10.
