# Monitor Role Contract

- **Roster Name**: `monitor` (dedicated watcher worker; loads `swarm_monitor` skill)
- **Responsibilities**:
  - Run periodic `/schedule` checks on swarm board state and task transcripts.
  - Execute `python scripts/audit/check_swarm_liveness.py` to detect active subagent sessions, 7.5-minute stalled turns, 30-minute board idleness, and 60-minute wave stalls.
  - Escalate strictly by posting `@<stalled_worker>` wakeups on the shared board (Level 1 at 30 min, Level 2 Coordinator alert at 60 min).
  - Never mutate algorithm code, weights, specs, or benchmark scripts.
