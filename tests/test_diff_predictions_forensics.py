# -*- coding: utf-8 -*-
"""Forensic diff, taxonomy predicate, FRSA-011 ratchet, truncation & blindfold tests."""
import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.diff_predictions import (
    DEFAULT_BASE_F1,
    DEFAULT_BASE_SUB,
    MAX_FORENSIC_MD_BYTES,
    check_promotion_ratchet,
    classify_failure_mode,
    main as diff_main,
    run_forensic_diff,
)


class TestDiffPredictionsForensics(unittest.TestCase):
    def test_01_promotion_ratchet_frsa011_anchor_and_regression_guard(self):
        self.assertEqual(DEFAULT_BASE_F1, 64.12)
        self.assertEqual(DEFAULT_BASE_SUB, 62.19)

        # Pass against FRSA-011 anchor (64.12, 62.19 -> comp 63.155 -> 63.16)
        r_pass = check_promotion_ratchet(64.50, 62.50)
        self.assertTrue(r_pass["promoted"])
        self.assertGreaterEqual(r_pass["d_comp"], 0.25)
        self.assertEqual(r_pass["eff_best_comp"], 63.16)

        # Pass with higher best_f1/best_sub within -0.50 tolerance
        r_tol = check_promotion_ratchet(65.60, 64.90, best_f1=66.00, best_sub=64.00)
        self.assertTrue(r_tol["promoted"])
        self.assertEqual(r_tol["eff_best_comp"], 65.0)
        self.assertEqual(r_tol["d_comp"], 0.25)

        # Regression guard: beats initial FRSA-011 anchor (+1.545) yet < eff_best_comp + 0.25 -> REJECTED
        r_reg = check_promotion_ratchet(65.60, 63.80, best_f1=66.00, best_sub=64.00)
        self.assertFalse(r_reg["promoted"])
        self.assertEqual(r_reg["eff_best_comp"], 65.0)
        self.assertAlmostEqual(r_reg["d_comp"], -0.30, places=2)
        self.assertIn("eff_best_comp 65.00", r_reg["reasons"][0])

        # Fail all three conditions
        r_fail_all = check_promotion_ratchet(63.00, 61.00, best_f1=66.00, best_sub=64.00)
        self.assertFalse(r_fail_all["promoted"])
        self.assertEqual(len(r_fail_all["reasons"]), 3)

    def test_02_taxonomy_priority_order_all_branches(self):
        self.assertEqual(classify_failure_mode("dựa vào hình ảnh", ["dựa vào hình ảnh"], 1.0, 1.0), "MATCHED")
        self.assertEqual(classify_failure_mode("lune", ["lune"], 0.995, 0.0), "MATCHED")
        self.assertEqual(classify_failure_mode("Dựa vào hình ảnh", ["dựa vào hình ảnh abc"], 0.2, 0.0), "PREAMBLE_ECHO")
        self.assertEqual(classify_failure_mode("", ["apple"], 0.0, 0.0), "TRUNCATION")
        self.assertEqual(classify_failure_mode("ab", ["hội an town"], 0.0, 0.0), "TRUNCATION")
        self.assertEqual(classify_failure_mode("trà", ["trà sữa trân châu"], 0.2, 0.0), "TRUNCATION")
        self.assertEqual(classify_failure_mode("ha noi cafe", ["hà nội cafê"], 0.5, 0.0), "GLYPH_CONFUSION")
        self.assertEqual(classify_failure_mode("oppo", ["hội an"], 0.0, 0.0), "PRIOR_HALLUCINATION")

    def test_03_forensic_diff_alias_missing_keys_and_8kb_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = [
                {"question_id": 1, "error_class": "CLASS_A", "is_calibration": True, "question": "Q_CAL_1"},
                {"question_id": 2, "error_class": "CLASS_B", "is_calibration": False, "question": "Q_HELD_SECRET"},
            ]
            m_path = os.path.join(tmp, "manifest.json")
            open(m_path, "w", encoding="utf-8").write(json.dumps(manifest))

            base_recs = [
                {"question_id": 1, "question": "Q_CAL_1", "gt_answers": ["apple"], "pred_answer": "oppo", "stage2_metrics": {"f1_verified": 0.0, "substring_verified": 0.0, "em_verified": 0.0}},
                {"question_id": 2, "question": "Q_HELD_SECRET", "gt_answers": ["secret_gt"], "pred_answer": "secret_base", "stage2_metrics": {"f1_verified": 1.0, "substring_verified": 1.0, "em_verified": 1.0}},
            ]
            cand_recs = [
                {"question_id": 1, "question": "Q_CAL_1", "gt_answers": ["apple"], "pred_answer": "apple", "stage2_metrics": {"f1_verified": 1.0, "substring_verified": 1.0, "em_verified": 1.0}},
                {"question_id": 2, "question": "Q_HELD_SECRET", "gt_answers": ["secret_gt"], "pred_answer": "secret_wrong", "stage2_metrics": {"f1_verified": 0.0, "substring_verified": 0.0, "em_verified": 0.0}},
            ]
            bp = os.path.join(tmp, "base.jsonl")
            cp = os.path.join(tmp, "cand.jsonl")
            open(bp, "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in base_recs) + "\n")
            open(cp, "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in cand_recs) + "\n")

            bsum = os.path.join(tmp, "base_sum.json")
            open(bsum, "w", encoding="utf-8").write(json.dumps({"benchmark_1222": {"token_f1_pct": 64.12, "substring_match_pct": 62.19}}))
            csum = os.path.join(tmp, "cand_sum.json")
            open(csum, "w", encoding="utf-8").write(json.dumps({"stage2": {"f1_verified_pct": 65.60, "substring_match_verified_pct": 63.80}, "performance": {"avg_latency_s": 1.75}}))
            prov = os.path.join(tmp, "prov.json")
            open(prov, "w", encoding="utf-8").write(json.dumps({"papers": [{"paper_id": "2501.00001", "status": "VERIFIED_TEX", "section": "Sec. 3"}]}))
            psum = os.path.join(tmp, "probe_sum.json")
            open(psum, "w", encoding="utf-8").write(json.dumps({"loo_decision": "FULL_PASS", "selected_mask": "1", "submask_hint": None, "full_comp_70": 42.10, "best_probe_70_comp": 40.92, "masks": {"1": {"comp_70": 42.10, "f1_70": 47.0, "sub_70": 37.2}}}))
            tel = os.path.join(tmp, "tel.json")
            open(tel, "w", encoding="utf-8").write(json.dumps({"peak_vram_gb": 9.4, "max_kv_diff": 0.0, "decoder_forward_calls_per_step_max": 1.0}))

            out_json = os.path.join(tmp, "forensic_delta.json")
            out_md = os.path.join(tmp, "forensic_delta.md")

            # Use --baseline alias and higher --best-f1/--best-sub so eff_best_comp = 65.00 displays in Anchor Floors
            rc = diff_main([
                "--candidate-preds", cp,
                "--candidate-summary", csum,
                "--baseline", bp,
                "--baseline-summary", bsum,
                "--probe-manifest", m_path,
                "--best-f1", "66.00",
                "--best-sub", "64.00",
                "--provenance-json", prov,
                "--probe-summary", psum,
                "--colab-telemetry", tel,
                "--max-kv-diff", "0.0",
                "--out-json", out_json,
                "--out-md", out_md,
            ])
            self.assertEqual(rc, 0)
            rep = json.load(open(out_json, "r", encoding="utf-8"))
            self.assertFalse(rep["ratchet"]["promoted"])
            self.assertEqual(rep["vs_baseline"], {"wins": 1, "losses": 1, "ties": 0})
            self.assertEqual(len(rep["calibration_diffs"]), 1)
            md_txt = open(out_md, "r", encoding="utf-8").read()
            self.assertIn("Comp >= 65.25%", md_txt)
            self.assertIn("2501.00001", md_txt)
            self.assertIn("FULL_PASS", md_txt)
            self.assertIn("max_kv_diff=0", md_txt)
            self.assertNotIn("Q_HELD_SECRET", md_txt)
            self.assertNotIn("secret_wrong", md_txt)

            # Missing required keys raise explicit errors rather than defaulting to 0.0
            bad_csum = os.path.join(tmp, "bad_csum.json")
            open(bad_csum, "w", encoding="utf-8").write(json.dumps({"stage2": {}}))
            with self.assertRaises(KeyError):
                run_forensic_diff(cp, candidate_summary_path=bad_csum, baseline_preds_path=bp, baseline_summary_path=bsum, probe_manifest_path=m_path)
            bad_bsum = os.path.join(tmp, "bad_bsum.json")
            open(bad_bsum, "w", encoding="utf-8").write(json.dumps({"benchmark_1222": {}}))
            with self.assertRaises(KeyError):
                run_forensic_diff(cp, candidate_summary_path=csum, baseline_preds_path=bp, baseline_summary_path=bad_bsum, probe_manifest_path=m_path)

            # 8 KB truncation test over 40 calibration QIDs (mixed unchanged + failing long strings)
            big_manifest = [{"question_id": 100 + i, "error_class": "CLASS_A", "is_calibration": True, "question": f"Q_{i}_" + ("x" * 120)} for i in range(40)]
            big_m_path = os.path.join(tmp, "big_m.json")
            open(big_m_path, "w", encoding="utf-8").write(json.dumps(big_manifest))
            big_base = [{"question_id": 100 + i, "question": big_manifest[i]["question"], "gt_answers": ["gt_" + ("g" * 50)], "pred_answer": "p_" + ("b" * 50), "stage2_metrics": {"f1_verified": 1.0 if i < 5 else 0.2, "substring_verified": 1.0, "em_verified": 0.0}} for i in range(40)]
            big_cand = [{"question_id": 100 + i, "question": big_manifest[i]["question"], "gt_answers": ["gt_" + ("g" * 50)], "pred_answer": "p_" + ("c" * 50), "stage2_metrics": {"f1_verified": 1.0 if i < 5 else 0.1, "substring_verified": 1.0, "em_verified": 0.0}} for i in range(40)]
            big_bp = os.path.join(tmp, "big_b.jsonl")
            big_cp = os.path.join(tmp, "big_c.jsonl")
            open(big_bp, "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in big_base) + "\n")
            open(big_cp, "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in big_cand) + "\n")
            trunc_md = os.path.join(tmp, "trunc.md")
            run_forensic_diff(big_cp, baseline_preds_path=big_bp, baseline_summary_path=None, probe_manifest_path=big_m_path, out_md_path=trunc_md)
            self.assertLessEqual(os.path.getsize(trunc_md), MAX_FORENSIC_MD_BYTES)
            self.assertIn("unchanged/omitted calibration QIDs", open(trunc_md, "r", encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
