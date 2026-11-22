#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-Python Google Drive API Uploader, Credential Checker & Marker Poller
(scripts/benchmark/colab_drive_sync.py).

Reads RCLONE_DRIVE_TOKEN_JSON from os.environ (never from CLI args and never
echoed/logged), constructs auto-refreshing OAuth2 Credentials when client_id
and client_secret are present, uploads files with exponential backoff on
transient HTTP errors (403/429/500/502/503), idempotently updates
drive_upload_receipt.json, verifies auto-refresh credentials, and polls
remote Drive folder markers.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CHUNK_BYTES = 1 << 20  # 1 MB streaming chunks
RESUMABLE_THRESHOLD_BYTES = 5 * (1 << 20)  # 5 MB
SKIP_HASH_NAMES = {"sha256sum.txt", "drive_upload_receipt.json"}


def write_streaming_sha256sum(local_dir: str) -> str:
    """Write POSIX sha256sum.txt over top-level regular files in local_dir."""
    manifest_path = os.path.join(local_dir, "sha256sum.txt")
    entries: List[str] = []
    for name in sorted(os.listdir(local_dir)):
        if name in SKIP_HASH_NAMES:
            continue
        fpath = os.path.join(local_dir, name)
        if not os.path.isfile(fpath):
            continue
        hasher = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
                hasher.update(chunk)
        entries.append(f"{hasher.hexdigest()}  {name}\n")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as mf:
        mf.writelines(entries)
    return manifest_path


def verify_refresh_creds(env_var: str = "RCLONE_DRIVE_TOKEN_JSON") -> Dict[str, Any]:
    """Validate auto-refreshing Drive OAuth credentials without printing secret values."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return {
            "valid": False,
            "env_var": env_var,
            "reason": f"Environment variable {env_var} is missing or empty",
            "has_client_id": False,
            "has_client_secret": False,
            "has_refresh_token": False,
            "has_access_token": False,
        }
    try:
        data = json.loads(raw)
    except Exception as exc:
        return {
            "valid": False,
            "env_var": env_var,
            "reason": f"Invalid JSON in {env_var}: {type(exc).__name__}",
            "has_client_id": False,
            "has_client_secret": False,
            "has_refresh_token": False,
            "has_access_token": False,
        }
    if not isinstance(data, dict):
        return {
            "valid": False,
            "env_var": env_var,
            "reason": f"{env_var} must decode to a JSON object",
            "has_client_id": False,
            "has_client_secret": False,
            "has_refresh_token": False,
            "has_access_token": False,
        }

    has_cid = bool(str(data.get("client_id") or "").strip())
    has_csec = bool(str(data.get("client_secret") or "").strip())
    has_ref = bool(str(data.get("refresh_token") or "").strip())
    has_acc = bool(str(data.get("access_token") or data.get("token") or "").strip())
    valid = has_cid and has_csec and has_ref
    reason = "ok" if valid else "Missing required auto-refresh fields (client_id, client_secret, refresh_token)"
    return {
        "valid": valid,
        "env_var": env_var,
        "reason": reason,
        "has_client_id": has_cid,
        "has_client_secret": has_csec,
        "has_refresh_token": has_ref,
        "has_access_token": has_acc,
    }


def build_drive_service(env_var: str = "RCLONE_DRIVE_TOKEN_JSON") -> Any:
    """Construct Drive v3 client from in-memory OAuth JSON env var."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        raise RuntimeError(f"Environment variable {env_var} is missing or empty")
    data = json.loads(raw)

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if data.get("client_id") and data.get("client_secret"):
        creds = Credentials(
            token=data.get("access_token") or data.get("token"),
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri") or "https://oauth2.googleapis.com/token",
            client_id=data["client_id"],
            client_secret=data["client_secret"],
        )
    else:
        creds = Credentials(token=data.get("access_token") or data["token"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _exec_retry(req_fn: Callable[[], Any], max_retries: int = 6) -> Any:
    try:
        from googleapiclient.errors import HttpError
    except ImportError:
        class HttpError(Exception):  # type: ignore[no-redef]
            resp = None

    for attempt in range(max_retries):
        try:
            return req_fn().execute(num_retries=3)
        except HttpError as exc:
            status = getattr(exc.resp, "status", 0)
            if status in (403, 429, 500, 502, 503) and attempt < max_retries - 1:
                time.sleep(min(2**attempt, 32))
                continue
            raise


def _ensure_folder_path(service: Any, folder_parts: Sequence[str], parent_id: Optional[str] = None) -> str:
    cur_parent = parent_id
    for part in folder_parts:
        safe_part = part.replace("'", "\\'")
        q = f"name='{safe_part}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if cur_parent:
            q += f" and '{cur_parent}' in parents"
        found = _exec_retry(
            lambda q_str=q: service.files().list(q=q_str, fields="files(id, name)", pageSize=5)
        ).get("files", [])
        if found:
            cur_parent = found[0]["id"]
        else:
            meta: Dict[str, Any] = {"name": part, "mimeType": "application/vnd.google-apps.folder"}
            if cur_parent:
                meta["parents"] = [cur_parent]
            cur_parent = _exec_retry(lambda m=meta: service.files().create(body=m, fields="id"))["id"]
    if not cur_parent:
        raise ValueError("folder_parts must not be empty")
    return cur_parent


def _resolve_folder_path_readonly(service: Any, folder_parts: Sequence[str]) -> Optional[str]:
    cur_parent: Optional[str] = None
    for part in folder_parts:
        safe_part = part.replace("'", "\\'")
        q = f"name='{safe_part}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if cur_parent:
            q += f" and '{cur_parent}' in parents"
        found = _exec_retry(
            lambda q_str=q: service.files().list(q=q_str, fields="files(id, name)", pageSize=5)
        ).get("files", [])
        if not found:
            return None
        cur_parent = found[0]["id"]
    return cur_parent


def _upload_single_dir(
    service: Any,
    src_dir: str,
    parent_id: str,
) -> Tuple[List[str], Optional[str]]:
    from googleapiclient.http import MediaFileUpload

    existing = {
        f["name"]: f["id"]
        for f in _exec_retry(
            lambda: service.files().list(
                q=f"'{parent_id}' in parents and trashed=false",
                fields="files(id, name, mimeType)",
                pageSize=1000,
            )
        ).get("files", [])
    }
    uploaded: List[str] = []
    for name in sorted(os.listdir(src_dir)):
        if name == "drive_upload_receipt.json":
            continue
        fpath = os.path.join(src_dir, name)
        if os.path.isdir(fpath):
            sub_id = _ensure_folder_path(service, [name], parent_id=parent_id)
            sub_files, _ = _upload_single_dir(service, fpath, sub_id)
            uploaded.extend([f"{name}/{sf}" for sf in sub_files])
            continue
        if not os.path.isfile(fpath):
            continue
        is_resumable = os.path.getsize(fpath) > RESUMABLE_THRESHOLD_BYTES

        def _make_req(p: str = fpath, n: str = name, res: bool = is_resumable) -> Any:
            media = MediaFileUpload(p, resumable=res)
            if n in existing:
                return service.files().update(fileId=existing[n], media_body=media, fields="id, name")
            return service.files().create(
                body={"name": n, "parents": [parent_id]},
                media_body=media,
                fields="id, name",
            )

        _exec_retry(_make_req)
        uploaded.append(name)
    return uploaded, existing.get("drive_upload_receipt.json")


def upload_dir_to_drive(
    local_dir: str,
    folder_parts: Sequence[str],
    service: Optional[Any] = None,
    write_sha256: bool = True,
) -> Dict[str, Any]:
    """Upload local_dir to Google Drive folder_parts and emit/update drive_upload_receipt.json."""
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Local directory not found: {local_dir}")
    if write_sha256:
        write_streaming_sha256sum(local_dir)

    drv = service or build_drive_service()
    folder_id = _ensure_folder_path(drv, folder_parts)
    uploaded_files, existing_receipt_id = _upload_single_dir(drv, local_dir, folder_id)

    receipt = {
        "folder_id": folder_id,
        "folder_path": "/".join(folder_parts),
        "uploaded_files": uploaded_files,
        "uploaded_count": len(uploaded_files),
        "uploaded_at_epoch": round(time.time(), 3),
    }
    receipt_path = os.path.join(local_dir, "drive_upload_receipt.json")
    with open(receipt_path, "w", encoding="utf-8", newline="\n") as rf:
        json.dump(receipt, rf, ensure_ascii=False, indent=2)

    from googleapiclient.http import MediaFileUpload

    def _receipt_req() -> Any:
        media = MediaFileUpload(receipt_path, resumable=False)
        if existing_receipt_id:
            return drv.files().update(fileId=existing_receipt_id, media_body=media, fields="id, name")
        return drv.files().create(
            body={"name": "drive_upload_receipt.json", "parents": [folder_id]},
            media_body=media,
            fields="id, name",
        )

    _exec_retry(_receipt_req)
    return receipt


def poll_markers(
    folder_id: Optional[str] = None,
    folder_path: Optional[str] = None,
    service: Optional[Any] = None,
) -> Dict[str, Any]:
    """Read-only poll of a Drive run folder for WAVE_COMPLETE_MARKER / WAVE_STALLED_MARKER."""
    drv = service or build_drive_service()
    resolved_id = folder_id
    if not resolved_id and folder_path:
        parts = [p for p in folder_path.strip("/").split("/") if p]
        resolved_id = _resolve_folder_path_readonly(drv, parts)
    if not resolved_id:
        return {
            "folder_id": None,
            "complete": False,
            "stalled": False,
            "files": [],
        }

    listed = _exec_retry(
        lambda: drv.files().list(
            q=f"'{resolved_id}' in parents and trashed=false",
            fields="files(id, name, size, mimeType)",
            pageSize=1000,
        )
    ).get("files", [])
    names = {f.get("name", "") for f in listed}
    return {
        "folder_id": resolved_id,
        "complete": "WAVE_COMPLETE_MARKER" in names,
        "stalled": "WAVE_STALLED_MARKER" in names,
        "files": [
            {
                "id": f.get("id"),
                "name": f.get("name"),
                "size": int(f.get("size", 0) or 0),
                "mimeType": f.get("mimeType"),
            }
            for f in sorted(listed, key=lambda x: str(x.get("name", "")))
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Popup-free pure-Python Google Drive uploader and poller.")
    parser.add_argument("--local-dir", required=False, default=None, help="Local directory on Colab VM to upload")
    parser.add_argument(
        "--folder",
        required=False,
        default=None,
        help="Slash-separated Drive folder path e.g. VLM_Track_Runs/track_1_iter_1",
    )
    parser.add_argument(
        "--write-sha256",
        action="store_true",
        default=False,
        help="Regenerate streaming sha256sum.txt before uploading",
    )
    parser.add_argument(
        "--verify-refresh-creds",
        action="store_true",
        default=False,
        help="Verify RCLONE_DRIVE_TOKEN_JSON includes client_id, client_secret, and refresh_token",
    )
    parser.add_argument(
        "--poll-markers",
        action="store_true",
        default=False,
        help="Poll a Drive folder for WAVE_COMPLETE_MARKER / WAVE_STALLED_MARKER",
    )
    parser.add_argument(
        "--folder-id",
        default=None,
        help="Drive folder ID for --poll-markers",
    )
    args = parser.parse_args(argv)

    if args.verify_refresh_creds:
        res = verify_refresh_creds()
        print(json.dumps(res, ensure_ascii=False))
        return 0 if res["valid"] else 1

    if args.poll_markers:
        if not args.folder_id and not args.folder:
            parser.error("--poll-markers requires --folder-id or --folder")
        res = poll_markers(folder_id=args.folder_id, folder_path=args.folder)
        print(json.dumps(res, ensure_ascii=False))
        return 0

    if not args.local_dir or not args.folder:
        parser.error("Upload mode requires --local-dir and --folder")

    parts = [p for p in args.folder.strip("/").split("/") if p]
    receipt = upload_dir_to_drive(args.local_dir, parts, write_sha256=args.write_sha256)
    print(
        f"[DRIVE_SYNC] Uploaded {receipt['uploaded_count']} files -> "
        f"{receipt['folder_path']} (folder_id={receipt['folder_id']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
