# Auditor Role Contract

- **Roster Name**: `auditor`
- **Workspace**: `inherit` (verifies git commits & machine-readable receipts)
- **Responsibilities**:
  - Run `python scripts/audit/verify_transcript_pdf_reads.py` before approving literature review.
  - Run `python scripts/audit/run_auditor_checks.py --mode pre-colab --commit-sha <SHA>` before any commit enters a Colab GPU wave (Checks #1–#6: immutable metrics diff, spec-binding LF SHA-256, bounded core/constants AST check, regex/string anti-hack check, local unit receipt, full test suite coverage).
  - Run `python scripts/audit/run_auditor_checks.py --mode post-colab --wave-dir <dir>` after Colab waves complete (Checks #7–#10: calibration bank size & diff audit, 9 output artifacts + `sha256sum -c`, held-out quarantine verification, anchored promotion ratchet).
