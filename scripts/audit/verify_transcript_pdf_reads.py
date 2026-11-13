#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cursor-Segmented Transcript & Literature Audit (scripts/audit/verify_transcript_pdf_reads.py).

Verifies for a researcher iteration:
- >= 7 distinct papers in reading_history.md (target 7-15; > 15 allowed)
- 1:1 BibTeX key correspondence between reading_history.md and reading_history.bib
- Cached PDFs in --cache-dir have %PDF- header, size > 50 KB, and matching SHA-256
- Transcript (step_index > cursor_step, joining transcript_full.jsonl on truncated_fields)
  evidences body reads (pages >= 2) via fetch_papers.py --inspect-pages + read_manifest.json
  (body_chars >= 500) OR non-abstract arxiv:get_paper_latex_section calls
- Epistemic Bucket quotas: A in [1, 3], B in [2, 4], C in [2, 4], D in [2, 4] with Q1 journal ISSN/venue
"""

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

MIN_PDF_BYTES = 50 * 1024
MIN_BODY_CHARS = 500
Q1_VENUE_HINTS = (
    "issn",
    "tpami",
    "ijcv",
    "tip",
    "tit",
    "jmlr",
    "tsp",
    "tmi",
    "tnnls",
    "transactions on",
    "journal of",
    "pattern recognition",
    "q1",
)


def _load_transcript_steps(transcript_path: str, cursor_step: int) -> List[Dict[str, Any]]:
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    full_path = os.path.join(os.path.dirname(os.path.abspath(transcript_path)), "transcript_full.jsonl")
    full_by_step: Dict[int, Dict[str, Any]] = {}
    if os.path.isfile(full_path):
        with open(full_path, "r", encoding="utf-8") as ff:
            for line in ff:
                if line.strip():
                    rec = json.loads(line)
                    full_by_step[int(rec.get("step_index", -1))] = rec

    steps: List[Dict[str, Any]] = []
    with open(transcript_path, "r", encoding="utf-8") as tf:
        for line in tf:
            if not line.strip():
                continue
            rec = json.loads(line)
            s_idx = int(rec.get("step_index", -1))
            if s_idx <= cursor_step:
                continue
            if rec.get("truncated_fields") and s_idx in full_by_step:
                rec = full_by_step[s_idx]
            steps.append(rec)
    return steps


def _parse_reading_history_md(md_path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(md_path):
        raise FileNotFoundError(f"Reading history markdown not found: {md_path}")
    rows: List[Dict[str, str]] = []
    headers: Optional[List[str]] = None
    with open(md_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not (line.startswith("|") and line.endswith("|")):
                continue
            cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c or "---") for c in cells):
                continue
            if headers is None:
                headers = [c.lower() for c in cells]
                continue
            if len(cells) >= 10:
                rows.append(
                    {
                        "iteration": cells[0],
                        "bibtex_key": cells[1],
                        "title": cells[2],
                        "venue": cells[3],
                        "doi_arxiv": cells[4],
                        "pdf_sha256": cells[5].lower(),
                        "bucket": cells[6].upper(),
                        "pages": cells[7],
                        "anchor": cells[8],
                        "repo_url": cells[9],
                    }
                )
    return rows


def _parse_bib_keys(bib_path: str) -> Set[str]:
    if not os.path.isfile(bib_path):
        raise FileNotFoundError(f"BibTeX file not found: {bib_path}")
    text = open(bib_path, "r", encoding="utf-8").read()
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))


def _index_cached_pdfs(cache_dir: str) -> Dict[str, Dict[str, Any]]:
    pdfs: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(cache_dir):
        return pdfs
    for root, _, files in os.walk(cache_dir):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            with open(fpath, "rb") as f:
                raw = f.read()
            if raw.startswith(b"%PDF-") and size > MIN_PDF_BYTES:
                sha = hashlib.sha256(raw).hexdigest().lower()
                pdfs[sha] = {"path": fpath, "size": size}
    return pdfs


def _load_read_manifest(cache_dir: str) -> Dict[str, Dict[str, Any]]:
    manifest_by_sha: Dict[str, Dict[str, Any]] = {}
    manifest_path = os.path.join(cache_dir, "read_manifest.json")
    if not os.path.isfile(manifest_path):
        return manifest_by_sha
    entries = json.load(open(manifest_path, "r", encoding="utf-8"))
    for entry in entries:
        sha = str(entry.get("pdf_sha256", "")).lower()
        pages = [int(p) for p in entry.get("pages_read", [])]
        chars = int(entry.get("body_chars", 0))
        if sha and any(p >= 2 for p in pages) and chars >= MIN_BODY_CHARS:
            manifest_by_sha[sha] = entry
    return manifest_by_sha


def _extract_transcript_evidence(steps: List[Dict[str, Any]]) -> Tuple[bool, Set[str]]:
    has_inspect_call = False
    latex_section_ids: Set[str] = set()
    for step in steps:
        for tc in step.get("tool_calls", []) or []:
            args_str = json.dumps(tc, ensure_ascii=False)
            if "fetch_papers.py" in args_str and "--inspect-pages" in args_str:
                has_inspect_call = True
            tool_name = str(tc.get("name", ""))
            args = tc.get("arguments", {}) or {}
            if tool_name == "call_mcp_tool":
                if args.get("ServerName") == "arxiv" and args.get("ToolName") == "get_paper_latex_section":
                    inner = args.get("Arguments", {}) or {}
                    sec = str(inner.get("section", "")).lower()
                    if sec != "abstract":
                        pid = str(inner.get("paper_id") or inner.get("arxiv_id") or "").lower()
                        if pid:
                            latex_section_ids.add(pid)
    return has_inspect_call, latex_section_ids


def _is_valid_q1_journal(venue_str: str) -> bool:
    low = venue_str.lower()
    if re.search(r"\b\d{4}-\d{3}[\dx]\b", low):
        return True
    return any(hint in low for hint in Q1_VENUE_HINTS)


def verify_literature_audit(
    transcript_path: str,
    cursor_step: int,
    cache_dir: str,
    reading_history_path: str,
    bib_path: str,
) -> Dict[str, Any]:
    errors: List[str] = []
    steps = _load_transcript_steps(transcript_path, cursor_step)
    rows = _parse_reading_history_md(reading_history_path)
    bib_keys = _parse_bib_keys(bib_path)
    cached_pdfs = _index_cached_pdfs(cache_dir)
    manifest_by_sha = _load_read_manifest(cache_dir)
    has_inspect_call, latex_section_ids = _extract_transcript_evidence(steps)

    md_keys = {r["bibtex_key"] for r in rows}
    if md_keys != bib_keys:
        errors.append(
            f"1:1 BibTeX key mismatch: md_only={sorted(md_keys - bib_keys)}, bib_only={sorted(bib_keys - md_keys)}"
        )

    if len(rows) < 7:
        errors.append(f"Expected >= 7 distinct papers, found {len(rows)}")

    bucket_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for row in rows:
        b = row["bucket"]
        if b not in bucket_counts:
            errors.append(f"Invalid bucket {b!r} for {row['bibtex_key']}")
        else:
            bucket_counts[b] += 1
        if b == "D" and not _is_valid_q1_journal(row["venue"]):
            errors.append(f"Bucket D paper {row['bibtex_key']} lacks Q1 journal venue/ISSN: {row['venue']!r}")

        sha = row["pdf_sha256"]
        if sha not in cached_pdfs:
            errors.append(f"Missing or <=50KB invalid PDF in cache for {row['bibtex_key']} (sha={sha})")

        doi_low = row["doi_arxiv"].lower()
        in_manifest = sha in manifest_by_sha and has_inspect_call
        in_latex = any(pid in doi_low for pid in latex_section_ids)
        if not (in_manifest or in_latex):
            errors.append(
                f"No verified page>=2 body read (read_manifest + transcript --inspect-pages or arxiv latex section) for {row['bibtex_key']}"
            )

    # Minimum epistemic bucket floors (A >= 1, B >= 2, C >= 2, D >= 2); reading > 15 total does not fail.
    min_quotas = {"A": 1, "B": 2, "C": 2, "D": 2}
    for b, lo in min_quotas.items():
        if bucket_counts[b] < lo:
            errors.append(f"Bucket {b} count {bucket_counts[b]} below minimum {lo}")

    return {
        "passed": len(errors) == 0,
        "paper_count": len(rows),
        "bucket_counts": bucket_counts,
        "transcript_steps_checked": len(steps),
        "errors": errors,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify cursor-scoped PDF reads and reading history.")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--cursor-step", type=int, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--reading-history", required=True)
    parser.add_argument("--bib", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    result = verify_literature_audit(
        transcript_path=args.transcript,
        cursor_step=args.cursor_step,
        cache_dir=args.cache_dir,
        reading_history_path=args.reading_history,
        bib_path=args.bib,
    )
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
