You are the Colab Launch micro-agent (< 30s) for Track {track_id}, Iteration {iter_k} (trace `{trace_id}`).

Follow `C:\Users\Admin\.gemini\config\skills\colab-workflow\SKILL.md` strictly:
1. Call `colab-mcp:open_colab_browser_connection` and `colab-mcp:get_cells(includeOutputs=false)`. Save the initial cell list JSON to `{initial_cells_path}`.
2. Add one `< 2s` launcher code cell (`add_code_cell`) and execute it (`run_code_cell`) to:
   - Read `HF_TOKEN` and `RCLONE_DRIVE_TOKEN` (fallback `RCLONE_DRIVE_TOKEN_JSON`) from `google.colab.userdata` in memory into `child_env` (NEVER print secrets, NEVER call `drive.mount()` or `notebook_login()`).
   - Ensure `/content/vlm_bench` clones/fetches `{staging_ref}` at `{commit_sha}`, copy `scripts/benchmark/run_colab_track_wave.sh` and `scripts/benchmark/colab_drive_sync.py` to `/tmp/`, verify `RCLONE_DRIVE_TOKEN_JSON` auto-refresh credentials, ensure Drive folder `{drive_folder}` exists and resolve its `run_folder_id`, and launch detached:
     `nohup bash /tmp/run_colab_track_wave.sh --track {track_id} --iter {iter_k} --attempt {attempt_n} --staging-ref {staging_ref} --commit-sha {commit_sha} --drive-folder {drive_folder} --best-probe-comp {best_probe_comp} > /tmp/track_nohup.log 2>&1 < /dev/null &`
   - Print ONLY `WAVE_ID=...`, `RUN_FOLDER_ID=...`, and `CONNECT_TS=...`.
3. Call `finish` immediately (do NOT poll or wait for the wave) with `{{"wave_id": "...", "run_folder_id": "...", "added_cell_ids": [...], "connect_ts": <float_epoch>}}`.
