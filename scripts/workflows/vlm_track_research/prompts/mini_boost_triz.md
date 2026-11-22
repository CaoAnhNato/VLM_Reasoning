You are running Tier-1 Mini-Boost TRIZ Contradiction Diagnosis after Iteration {iter_k} failed to improve (`consecutive_no_gain == 1`) on Track {track_id} ({track_name}).

Inputs:
1. Epistemic Preamble (< 3 KB):
{epistemic_preamble}

2. Iteration {iter_k} Spec (`{spec_path}`):
{spec_excerpt}

3. Latest Forensic Delta (`{forensic_md_path}`):
{forensic_excerpt}

Apply TRIZ contradiction analysis: identify the physical/mathematical contradiction between what improved and what regressed (e.g., calibration sharpness vs. off-target amplification or KV invariance), state the root cause, formulate one concise `dead_end_rule` to avoid repeating this failure, and give a concrete `next_direction_hint` for Iteration {next_iter_k}.
Call `finish` with `{{"contradiction": "...", "root_cause": "...", "dead_end_rule": "...", "next_direction_hint": "..."}}`.
