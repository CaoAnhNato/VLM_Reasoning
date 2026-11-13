# Researcher Role Contract

- **Roster Name Pattern**: `researcher-t<team>`
- **Workspace**: `inherit` (read-only research; writes ONLY to team report/spec paths)
- **Responsibilities**:
  - Execute deep literature survey (>= 8 unique PDFs across Buckets A–D: >=1 A, >=2 B, >=2 C, >=2 D) via `tools/s2_folks_cli/fetch_papers.py --inspect-pages 2-7` (or `arxiv` MCP `get_paper_latex_section` for raw TeX sources).
  - Record every inspected paper 1:1 in `reports/reading_history_team<T>.md` and `reports/refs_team<T>.bib` adhering to `reports/READING_HISTORY_TEMPLATE.md`.
  - Author mathematically complete `docs/specs/spec_team<T>_iter<K>.md` adhering strictly to `docs/specs/SPEC_TEMPLATE.md` before `implementer` touches code.
