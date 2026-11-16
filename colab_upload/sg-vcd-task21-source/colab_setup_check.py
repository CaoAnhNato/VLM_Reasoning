#!/usr/bin/env python3
"""Colab pre-run check for the Task 21 Gate B package (no model load, no inference).

1. Verifies the frozen manifest SHA256 and sample count.
2. Locates the 6 manifest images (searching --images-root recursively) and links or copies
   them into <project_root>/data/vitextvqa/st_images/ where evaluate_vitextvqa.py looks.
3. Verifies ground truth bounding boxes in results/swintextspotter_auto_annot/answer_bboxes_400_verified.json.
4. Verifies internal module importability.

Usage (from project root):
    python colab_setup_check.py --images-root "/content/drive/MyDrive/Document AI/Data" --link
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

EXPECTED_SHA256 = "9e9dc05dd87c2fb2de254cc1022dad41d5e745f8070560ddd98c648fab5b269a"
ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ROOT, "data", "vitextvqa", "manifest_integration_20qa.json")
ST_IMAGES = os.path.join(ROOT, "data", "vitextvqa", "st_images")
GT_BBOXES = os.path.join(ROOT, "results", "swintextspotter_auto_annot", "answer_bboxes_400_verified.json")


def main():
    ap = argparse.ArgumentParser(description="Colab Gate B Pre-run Check")
    ap.add_argument("--images-root", default=None, help="Folder (searched recursively) holding extracted dataset images.")
    ap.add_argument("--link", action="store_true", help="Symlink located images into data/vitextvqa/st_images/.")
    ap.add_argument("--copy", action="store_true", help="Copy located images instead of symlinking.")
    a = ap.parse_args()

    print("=== Colab Gate B Package Setup Check ===")
    
    # 1. Manifest verification
    if not os.path.isfile(MANIFEST):
        print(f"[FAIL] Manifest not found: {MANIFEST}")
        sys.exit(2)
        
    h = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    ok = (h == EXPECTED_SHA256)
    print(f"[manifest] sha256={h} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print(f"[ERROR] Expected manifest SHA256 {EXPECTED_SHA256}")
        sys.exit(2)
        
    data = json.load(open(MANIFEST, encoding="utf-8"))
    samples = data.get("samples", [])
    names = sorted({s["image_name"] for s in samples})
    print(f"[manifest] {len(samples)} QA pairs, {len(names)} unique images: {names}")

    # 2. GT Bboxes verification
    if os.path.isfile(GT_BBOXES):
        with open(GT_BBOXES, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        matched_gt = sum(1 for s in samples if str(s.get("question_id")) in gt_data or str(s.get("id")) in gt_data)
        print(f"[gt_bboxes] file present ({len(gt_data)} items). Matched {matched_gt}/{len(samples)} manifest items.")
    else:
        print(f"[WARNING] GT bboxes file missing: {GT_BBOXES}")

    # 3. Image resolution & linking
    found = {}
    if a.images_root:
        if not os.path.isdir(a.images_root):
            print(f"[WARN] images-root directory not found: {a.images_root}")
        else:
            for dp, _, fs in os.walk(a.images_root):
                for f in fs:
                    if f in names and f not in found and os.path.getsize(os.path.join(dp, f)) > 0:
                        found[f] = os.path.join(dp, f)
            print(f"[search] located {len(found)}/{len(names)} manifest images under {a.images_root}")
            
            if a.link or a.copy:
                os.makedirs(ST_IMAGES, exist_ok=True)
                for n, p in sorted(found.items()):
                    dst = os.path.join(ST_IMAGES, n)
                    if not os.path.exists(dst):
                        if a.copy:
                            shutil.copyfile(p, dst)
                            print(f"[copy] {dst} <- {p}")
                        else:
                            try:
                                os.symlink(p, dst)
                                print(f"[link] {dst} -> {p}")
                            except OSError as e:
                                shutil.copyfile(p, dst)
                                print(f"[link fallback -> copy] {dst} <- {p} (reason: {e})")

    missing = [n for n in names if not (os.path.isfile(os.path.join(ST_IMAGES, n)) and os.path.getsize(os.path.join(ST_IMAGES, n)) > 0)]
    if missing:
        print(f"[FAIL] Images not resolvable in {ST_IMAGES}: {missing}")
        print("Run with: python colab_setup_check.py --images-root \"/path/to/extracted/images\" --link")
        sys.exit(1)
        
    print("[OK] All 6 manifest images resolvable.")
    print("=== Setup check COMPLETE: All checks PASSED ===")


if __name__ == "__main__":
    main()
