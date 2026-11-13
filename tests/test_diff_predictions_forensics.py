# -*- coding: utf-8 -*-
"""Forensic diff, taxonomy predicate, and 100% ratchet branch coverage tests."""
import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.diff_predictions import (
    check_promotion_ratchet,
    classify_failure_mode,
    main as diff_main,
    run_forensic_diff,
)


class TestDiffPredictionsForensics(unittest.TestCase):
    def test_01_promotion_ratchet_all_branches(self):
        # Pass against Iter 23 anchor (60.19, 61.46 -> comp 60.825)
        r_pass = check_promotion_ratchet(60.60, 61.70)
        self.assertTrue(r_pass["promoted"])
        self.assertGreaterEqual(r_pass["d_comp"], 0.25)

        # Pass with higher best_f1/best_sub within -0.50 tolerance
        r_tol = check_promotion_ratchet(61.00, 61.60, best_f1=61.40, best_sub=62.00)
        self.assertTrue(r_tol["promoted"])

        # Fail all three conditions: d_comp < +0.25, f1 < floor, sub < floor
        r_fail_all = check_promotion_ratchet(60.00, 61.20, best_f1=61.00, best_sub=62.00)
        self.assertFalse(r_fail_all["promoted"])
        self.assertEqual(len(r_fail_all["reasons"]), 3)

        # Fail only d_comp (meets anchor floors)
        r_fail_d = check_promotion_ratchet(60.20, 61.50)
        self.assertFalse(r_fail_d["promoted"])
        self.assertEqual(len(r_fail_d["reasons"]), 1)

    def test_02_taxonomy_priority_order_all_branches(self):
        # Priority 1: MATCHED
        self.assertEqual(classify_failure_mode("dựa vào hình ảnh", ["dựa vào hình ảnh"], 1.0, 1.0), "MATCHED")
        self.assertEqual(classify_failure_mode("lune", ["lune"], 0.995, 0.0), "MATCHED")

        # Priority 2: PREAMBLE_ECHO (outranks truncation & glyph)
        self.assertEqual(classify_failure_mode("Dựa vào hình ảnh", ["dựa vào hình ảnh abc"], 0.2, 0.0), "PREAMBLE_ECHO")

        # Priority 3: TRUNCATION (empty, tiny vs long GT, prefix)
        self.assertEqual(classify_failure_mode("", ["apple"], 0.0, 0.0), "TRUNCATION")
        self.assertEqual(classify_failure_mode("ab", ["hội an town"], 0.0, 0.0), "TRUNCATION")
        self.assertEqual(classify_failure_mode("trà", ["trà sữa trân châu"], 0.2, 0.0), "TRUNCATION")

        # Priority 4: GLYPH_CONFUSION (diacritic/OCR near match)
        self.assertEqual(classify_failure_mode("ha noi cafe", ["hà nội cafê"], 0.5, 0.0), "GLYPH_CONFUSION")

        # Priority 5: PRIOR_HALLUCINATION
        self.assertEqual(classify_failure_mode("oppo", ["hội an"], 0.0, 0.0), "PRIOR_HALLUCINATION")

    def test_03_forensic_diff_calibration_isolation_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = [
                {"question_id": 1, "error_class": "CLASS_A", "is_calibration": True, "question": "Q_CAL"},
                {"question_id": 2, "error_class": "CLASS_B", "is_calibration": False, "question": "Q_HELD_SECRET"},
            ]
            m_path = os.path.join(tmp, "manifest.json")
            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            base_recs = [
                {"question_id": 1, "question": "Q_CAL", "gt_answers": ["apple"], "pred_answer": "oppo", "stage2_metrics": {"f1_verified": 0.0, "substring_verified": 0.0, "em_verified": 0.0}},
                {"question_id": 2, "question": "Q_HELD_SECRET", "gt_answers": ["secret_gt"], "pred_answer": "secret_base", "stage2_metrics": {"f1_verified": 1.0, "substring_verified": 1.0, "em_verified": 1.0}},
            ]
            cand_recs = [
                {"question_id": 1, "question": "Q_CAL", "gt_answers": ["apple"], "pred_answer": "apple", "stage2_metrics": {"f1_verified": 1.0, "substring_verified": 1.0, "em_verified": 1.0}},
                {"question_id": 2, "question": "Q_HELD_SECRET", "gt_answers": ["secret_gt"], "pred_answer": "secret_wrong", "stage2_metrics": {"f1_verified": 0.0, "substring_verified": 0.0, "em_verified": 0.0}},
            ]
            bp = os.path.join(tmp, "base.jsonl")
            cp = os.path.join(tmp, "cand.jsonl")
            open(bp, "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in base_recs) + "\n")
            open(cp, "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in cand_recs) + "\n")

            bsum = os.path.join(tmp, "base_sum.json")
            open(bsum, "w", encoding="utf-8").write(json.dumps({"benchmark_1222": {"token_f1_pct": 60.19, "substring_match_pct": 61.46}}))

            out_json = os.path.join(tmp, "forensic_delta.json")
            out_md = os.path.join(tmp, "forensic_delta.md")

            rc = diff_main([
                "--candidate-preds", cp,
                "--baseline-preds", bp,
                "--baseline-summary", bsum,
                "--probe-manifest", m_path,
                "--out-json", out_json,
                "--out-md", out_md,
            ])
            self.assertEqual(rc, 0)
            rep = json.load(open(out_json, "r", encoding="utf-8"))
            self.assertEqual(rep["vs_baseline"], {"wins": 1, "losses": 1, "ties": 0})
            self.assertEqual(len(rep["calibration_diffs"]), 1)
            self.assertEqual(rep["calibration_diffs"][0]["question_id"], 1)
            md_txt = open(out_md, "r", encoding="utf-8").read()
            self.assertNotIn("Q_HELD_SECRET", md_txt)
            self.assertNotIn("secret_wrong", md_txt)


if __name__ == "__main__":
    unittest.main()
