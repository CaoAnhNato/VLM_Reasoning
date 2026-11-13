# Coordinator Role Contract

- **Roster Name**: `coordinator` (root parent conversation)
- **Responsibilities**:
  - Maintain campaign budgets (`max_teams=3`, `max_iterations=8`, hard stopping after 2 consecutive flat waves).
  - Dispatch research, critique, implementation, audit, and batched Colab GPU waves via native `/swarm` (`add`, `status`, `model`, `kill`, `title`).
  - Adjudicate cross-team conflicts, assign unique non-overlapping file paths or `workspace="share"` per team, and merge promoted team commits back onto `swarm/base` or team champion branches.
  - Enforce that held-out QIDs (70 probe + non-calibration 400 benchmark) are NEVER opened at sample level by any agent.
