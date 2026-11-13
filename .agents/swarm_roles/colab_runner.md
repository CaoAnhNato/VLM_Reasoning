# Colab Runner Role Contract

- **Roster Name**: `colab_runner`
- **Workspace**: `inherit`
- **Responsibilities**:
  - Strictly follow `.agents/rules/colab-mcp-rules.md` and `/colab-workflow`: call `open_colab_browser_connection` once, snapshot `get_cells()` before modifying, decouple long runs (`> 3 min`) via `nohup bash /tmp/run_colab_swarm_wave.sh ... > /tmp/wave.log 2>&1 &`, monitor via `schedule` timers without blocking or tight polling loops.
  - Verify Drive sync, `sha256sum.txt`, and `WAVE_COMPLETE_MARKER` before downloading artifacts locally (`forensic_delta`, `summary_metrics`, `probe_ablation_summary.json`, `data/local_calibration_bank/`).
  - ALWAYS delete ONLY runner-added cells and execute `from google.colab import runtime; runtime.unassign()` synchronously once the wave finishes or stalls.
