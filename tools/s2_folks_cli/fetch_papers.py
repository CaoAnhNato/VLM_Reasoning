#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rate-Locked Semantic Scholar (allenai/s2-folks) Retrieval & PDF Body Inspection CLI.

Enforces:
- Win32 msvcrt.locking / POSIX fcntl.flock mutex at ~/.gemini/s2_api.lock (<= 90s)
- >= 1.1s inter-request spacing, 5 exponential-backoff retries on HTTP 429/504
- Strict https:// TLS verification (verify=True), connect/read timeout=(10, 30), <= 50 MB stream cap
- SEMANTIC_SCHOLAR_API_KEY / S2_API_KEY lookup (omits X-API-KEY header when empty)
- %PDF- magic header + > 50 KB size validation
- Bounded --inspect-pages 2-7 extraction (<= 6 pages, <= 20 KB) logging read_manifest.json
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from pypdf import PdfReader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_S2_LOCK = os.path.join(os.path.expanduser("~"), ".gemini", "s2_api.lock")
DEFAULT_CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".gemini", "vlm_paper_cache")
S2_GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = (
    "paperId,title,year,venue,publicationTypes,journal,"
    "externalIds,openAccessPdf,citationStyles,authors"
)
MAX_PDF_BYTES = 50 * 1024 * 1024
MIN_PDF_BYTES = 50 * 1024
MAX_INSPECT_PAGES = 6
MAX_INSPECT_BYTES = 20 * 1024


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned.strip("_") or "paper"


def _validate_https_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(f"Only https:// URLs are permitted, got: {url!r}")


@contextmanager
def _acquire_s2_lock(lock_path: str = DEFAULT_S2_LOCK, timeout_s: float = 90.0) -> Iterator[int]:
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    locked = False
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for S2 API lock {lock_path}")
                time.sleep(0.05)
        yield fd
    finally:
        if locked:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


class S2RateLimitedClient:
    """Thread/process-safe rate-locked client adapting allenai/s2-folks HTTP patterns."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        lock_path: str = DEFAULT_S2_LOCK,
        min_spacing_s: float = 1.1,
        lock_timeout_s: float = 90.0,
        max_retries: int = 5,
        session: Optional[requests.Session] = None,
    ) -> None:
        resolved_key = (
            api_key
            if api_key is not None
            else (os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY") or "")
        ).strip()
        self.api_key = resolved_key
        self.lock_path = lock_path
        self.stamp_path = f"{lock_path}.last_ts"
        self.min_spacing_s = float(min_spacing_s)
        self.lock_timeout_s = float(lock_timeout_s)
        self.max_retries = int(max_retries)
        self.session = session or requests.Session()

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "vlm-swarm-s2-cli/1.0"}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        return headers

    def _respect_spacing(self) -> None:
        now = time.time()
        last_ts = 0.0
        if os.path.isfile(self.stamp_path):
            try:
                with open(self.stamp_path, "r", encoding="utf-8") as f:
                    last_ts = float(f.read().strip() or "0")
            except (OSError, ValueError):
                last_ts = 0.0
        wait_s = self.min_spacing_s - (now - last_ts)
        if wait_s > 0:
            time.sleep(wait_s)
        with open(self.stamp_path, "w", encoding="utf-8") as f:
            f.write(f"{time.time():.6f}")

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _validate_https_url(url)
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            with _acquire_s2_lock(self.lock_path, timeout_s=self.lock_timeout_s):
                self._respect_spacing()
                resp = self.session.get(
                    url,
                    params=params,
                    headers=self._build_headers(),
                    timeout=(10, 30),
                    verify=True,
                )
            if resp.status_code in (429, 502, 503, 504):
                backoff = min(30.0, (2 ** attempt) * self.min_spacing_s)
                last_exc = requests.HTTPError(f"HTTP {resp.status_code} from {url}", response=resp)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp.json()
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Failed to query {url}")

    def download_pdf(self, pdf_url: str, dest_path: str) -> Tuple[str, int]:
        _validate_https_url(pdf_url)
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        tmp_path = f"{dest_path}.part.{os.getpid()}"
        total = 0
        sha = hashlib.sha256()
        with _acquire_s2_lock(self.lock_path, timeout_s=self.lock_timeout_s):
            self._respect_spacing()
            with self.session.get(
                pdf_url,
                headers={"User-Agent": "vlm-swarm-s2-cli/1.0"},
                stream=True,
                timeout=(10, 30),
                verify=True,
            ) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as out:
                    first_chunk = True
                    for chunk in resp.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        if first_chunk:
                            if not chunk.startswith(b"%PDF-"):
                                raise ValueError(f"Response from {pdf_url} lacks %PDF- magic header")
                            first_chunk = False
                        total += len(chunk)
                        if total > MAX_PDF_BYTES:
                            raise ValueError(f"PDF from {pdf_url} exceeds {MAX_PDF_BYTES} byte limit")
                        sha.update(chunk)
                        out.write(chunk)
        if total <= MIN_PDF_BYTES:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise ValueError(f"Downloaded PDF {pdf_url} is <= 50 KB ({total} bytes)")
        os.replace(tmp_path, dest_path)
        return sha.hexdigest(), total


def search_and_download_paper(
    query: str,
    team: int,
    iter_k: int,
    limit: int = 5,
    cache_root: str = DEFAULT_CACHE_ROOT,
    client: Optional[S2RateLimitedClient] = None,
) -> List[Dict[str, Any]]:
    """Adapt search_bulk + get_open_access_pdf/simple.py to cache valid PDFs locally."""
    s2 = client or S2RateLimitedClient()
    url = f"{S2_GRAPH_BASE}/paper/search"
    data = s2.get_json(url, params={"query": query, "limit": limit, "fields": DEFAULT_FIELDS})
    papers = data.get("data", []) or []

    cache_dir = os.path.join(cache_root, f"team_{team}", f"iter_{iter_k}")
    os.makedirs(cache_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for paper in papers:
        paper_id = str(paper.get("paperId") or "unknown")
        oa = paper.get("openAccessPdf") or {}
        pdf_url = oa.get("url") if isinstance(oa, dict) else None
        cached_pdf: Optional[str] = None
        pdf_sha256: Optional[str] = None
        pdf_size: Optional[int] = None
        if pdf_url and str(pdf_url).startswith("https://"):
            dest = os.path.join(cache_dir, f"{_sanitize_filename(paper_id)}.pdf")
            try:
                pdf_sha256, pdf_size = s2.download_pdf(pdf_url, dest)
                cached_pdf = dest
            except Exception as exc:
                paper["download_error"] = str(exc)
        results.append(
            {
                "paperId": paper_id,
                "title": paper.get("title"),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "journal": paper.get("journal"),
                "externalIds": paper.get("externalIds"),
                "openAccessPdf": oa,
                "citationStyles": paper.get("citationStyles"),
                "cached_pdf": cached_pdf,
                "pdf_sha256": pdf_sha256,
                "pdf_size": pdf_size,
            }
        )
    return results


def fetch_reference_hop(
    paper_id: str,
    limit: int = 25,
    client: Optional[S2RateLimitedClient] = None,
) -> List[Dict[str, Any]]:
    """Adapt self_citations_on_a_paper/main.py for 1-hop reference expansion."""
    s2 = client or S2RateLimitedClient()
    ref_fields = (
        "citedPaper.paperId,citedPaper.title,citedPaper.year,citedPaper.venue,"
        "citedPaper.publicationTypes,citedPaper.journal,citedPaper.externalIds,"
        "citedPaper.openAccessPdf,citedPaper.citationStyles,citedPaper.authors"
    )
    url = f"{S2_GRAPH_BASE}/paper/{paper_id}/references"
    data = s2.get_json(url, params={"fields": ref_fields, "limit": limit})
    refs = []
    for row in data.get("data", []) or []:
        cited = row.get("citedPaper") or {}
        if cited:
            refs.append(cited)
    return refs


def inspect_pdf_pages(
    pdf_path: str,
    page_range: str = "2-7",
    team: int = 0,
    iter_k: int = 0,
    paper_id: Optional[str] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract bounded body text (<= 6 pages, <= 20 KB) from cached PDF and append to read_manifest.json.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    file_size = os.path.getsize(pdf_path)
    if file_size <= MIN_PDF_BYTES:
        raise ValueError(f"PDF {pdf_path} must be > 50 KB, got {file_size} bytes")

    with open(pdf_path, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"%PDF-"):
        raise ValueError(f"Invalid PDF magic header in {pdf_path}")
    pdf_sha256 = hashlib.sha256(raw).hexdigest()

    parts = page_range.split("-")
    start_p = int(parts[0])
    end_p = int(parts[1]) if len(parts) > 1 else start_p
    if start_p < 1 or end_p < start_p or (end_p - start_p + 1) > MAX_INSPECT_PAGES:
        raise ValueError(
            f"page_range must select 1..{MAX_INSPECT_PAGES} pages (1-indexed), got {page_range!r}"
        )

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    pages_read: List[int] = []
    chunks: List[str] = []
    for p_num in range(start_p, min(end_p, total_pages) + 1):
        pages_read.append(p_num)
        txt = reader.pages[p_num - 1].extract_text() or ""
        chunks.append(f"--- PAGE {p_num} ---\n{txt.strip()}")

    combined = "\n\n".join(chunks)
    encoded = combined.encode("utf-8")
    if len(encoded) > MAX_INSPECT_BYTES:
        combined = encoded[:MAX_INSPECT_BYTES].decode("utf-8", errors="ignore")

    resolved_manifest = manifest_path or os.path.join(
        os.path.dirname(os.path.abspath(pdf_path)), "read_manifest.json"
    )
    entries: List[Dict[str, Any]] = []
    if os.path.isfile(resolved_manifest):
        try:
            with open(resolved_manifest, "r", encoding="utf-8") as mf:
                loaded = json.load(mf)
                if isinstance(loaded, list):
                    entries = loaded
        except Exception:
            entries = []

    record = {
        "team": int(team),
        "iter": int(iter_k),
        "paper_id": paper_id or os.path.splitext(os.path.basename(pdf_path))[0],
        "pdf_path": os.path.abspath(pdf_path),
        "pdf_sha256": pdf_sha256,
        "pages_read": pages_read,
        "body_chars": len(combined),
    }
    entries.append(record)
    os.makedirs(os.path.dirname(os.path.abspath(resolved_manifest)), exist_ok=True)
    tmp_m = f"{resolved_manifest}.tmp.{os.getpid()}"
    with open(tmp_m, "w", encoding="utf-8") as mf:
        json.dump(entries, mf, ensure_ascii=False, indent=2)
    os.replace(tmp_m, resolved_manifest)

    return {"manifest_record": record, "text": combined}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rate-locked Semantic Scholar & PDF reader CLI")
    parser.add_argument("--query", type=str, default=None, help="Search S2 and download OA PDFs")
    parser.add_argument("--references", type=str, default=None, help="Fetch 1-hop references for S2 paperId")
    parser.add_argument("--inspect-pages", type=str, default=None, help="1-indexed page range e.g. 2-7")
    parser.add_argument("--pdf", type=str, default=None, help="Local PDF path for --inspect-pages")
    parser.add_argument("--paper-id", type=str, default=None)
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--iter", dest="iter_k", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args(argv)

    if args.inspect_pages:
        if not args.pdf:
            parser.error("--pdf is required with --inspect-pages")
        out = inspect_pdf_pages(
            pdf_path=args.pdf,
            page_range=args.inspect_pages,
            team=args.team,
            iter_k=args.iter_k,
            paper_id=args.paper_id,
        )
        print(json.dumps(out["manifest_record"], ensure_ascii=False, indent=2))
        print(out["text"])
        return 0

    if args.references:
        refs = fetch_reference_hop(args.references, limit=args.limit)
        print(json.dumps(refs, ensure_ascii=False, indent=2))
        return 0

    if args.query:
        papers = search_and_download_paper(
            query=args.query,
            team=args.team,
            iter_k=args.iter_k,
            limit=args.limit,
            cache_root=args.cache_dir,
        )
        print(json.dumps(papers, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
