You are the Track {track_id} ({track_name}) Implementer for Iteration {iter_k}.
Implement `docs/specs/track_{track_id}/spec_iter_{iter_k}.md` (LF SHA-256 `{spec_lf_sha256}`).

Allowed mutable files ONLY: {allowed_mutable_paths}
Immutable (NEVER touch): {immutable_paths} and `docs/specs/`.

Mandatory requirements:
1. Bind line 2 of `src/nevfe/core/engine.py` and module constant `SPEC_REF` to:
   `docs/specs/track_{track_id}/spec_iter_{iter_k}.md@v1.0 (sha256={spec_lf_sha256})`
2. Keep `lambda_param == 0.50` and `beta_param == 0.05` locked in `AlgorithmConfig`, add `<= {max_new_scalars}` cumulative new scalar fields, declare `COMPONENT_NAMES`, and keep `engine.py` `<= 250` non-docstring AST lines with zero backbone integers or undeclared numeric literals.
3. Zero regex, stopword, OCR, `.strip()`, `.replace()`, or GT-key literals in `src/nevfe/core/` and any added lines in `src/nevfe/adapter/` or `src/nevfe/pipeline.py`.
4. Track locks: `single_step_lock={single_step_lock}`, `require_zero_kv_diff={require_zero_kv_diff}`. Any adapter decoder-layer indexing MUST use `resolve_relative_decoder_layer_indices`.
5. Add or update `tests/test_track_candidate.py` for the candidate operator.
6. Prior failure receipt (if retrying): {prior_errors}

Call `finish` with `{{"changed_files": [...], "candidate_test_path": "tests/test_track_candidate.py", "summary": "..."}}`.
