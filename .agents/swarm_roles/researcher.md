# Researcher Role Contract (`@researcher-1`, `@researcher-2`, `@researcher-3`)

- **Model / Workspace**: `gemini-4-carbon-high` | `workspace:"share"` on dedicated branch `swarm/team-<i>` (`i in {1, 2, 3}`, rooted at `swarm/base`; NEVER commit to `main`, `iter23_staging`, or `swarm/base`, and NEVER push to `origin`).
- **Disjoint Physical Domains**:
  - `researcher-1` (`swarm/team-1`): Input Optical & Native RGB Sensing (transient per-question RGB sub-crops; `self._cached_vit_embeds` stays immutable).
  - `researcher-2` (`swarm/team-2`): Autoregressive Trajectory & Output Distribution (`t=1` vs `t>=2` multi-step simplex/free-energy trajectory dynamics).
  - `researcher-3` (`swarm/team-3`): Intra-Decoder Cross-Modal Routing & Latent Subspace (`Qwen2.5-0.5B` layers `1..24`, `14:2` `7:1` GQA routing & residual subspace).
- **Per-Iteration Protocol (`k = 1..10`)**:
  1. Post a 4-line Hypothesis Claim on the `/swarm` board.
  2. Discover & read body pages `>= 2` of **`7-15` distinct PDFs** (`> 50 KB`, `%PDF-`) across Buckets `A (1-3)`, `B (2-4)`, `C (2-4)`, `D (2-4 Q1 journal with ISSN)` into `C:/Users/Admin/.gemini/vlm_paper_cache/team_<i>/iter_<k>/`. Route **100%** of Semantic Scholar calls through `C:\ProgramData\anaconda3\python.exe tools/s2_folks_cli/fetch_papers.py` (`--query` / `--references` / `--inspect-pages 2-7 --pdf <path> --team <i> --iter <k>`); **NEVER** call MCP `semantic-scholar` directly. Full access to `arxiv`, `keenable`, `keenable-select`, `github`, `octocode` MCPs.
  3. Run `< 0.5s` CPU open-loop inspection on `data/local_calibration_bank/` via `scripts/local/run_guarded.py --idle-timeout-s 1800` (`CUDA_VISIBLE_DEVICES=""`). Never open held-out non-calibration QIDs at sample level.
  4. Author unified `<= 3`-parameter operator (ZERO multi-paper formula stitching!) in `docs/specs/team_<i>/spec_iter_<k>.md` (`v1.0`, matching `docs/specs/SPEC_TEMPLATE.md`) and `1:1` aligned `reports/team_<i>/direction_<d>/reading_history.{md,bib}` (`reports/READING_HISTORY_TEMPLATE.md`), then `@critic` for Structural Independence review.
  5. Once `@critic` clears `v1.0`, commit onto `swarm/team-<i>`, compute LF-normalized `SPEC_SHA256 = sha256(open(spec,'rb').read().replace(b'\r\n',b'\n')).hexdigest()`, and hand `[spec_commit_sha]` + `SPEC_SHA256` to `@impl-<i>`.
  6. Handle `@impl-<i>` `SPEC_DEFECT_REPORT` with mechanical `v1.1..v1.3` updates (re-clearing `@critic` if any governing equation or parameter changes). On post-Colab failure, execute path-scoped rollback `git checkout <team_best_commit> -- src/nevfe/ tests/` (preserving `reports/team_<i>/`), pivoting direction after 3 strikes, and authoring `reports/team_<i>/FINAL_TEAM_SYNTHESIS.md` at Iteration 10.
