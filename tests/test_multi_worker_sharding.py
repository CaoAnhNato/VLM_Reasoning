# -*- coding: utf-8 -*-
"""
Unit tests for Multi-Worker Sharding Engine and Stage 1/2 Algorithmic Upgrades (Milestone R2-1).
Verifies:
1. Deterministic data sharding across workers (zero overlap, zero dropped samples).
2. Per-shard checkpointing and predictions output generation.
3. Atomic shard aggregation (merging, sorting by sample ID, master artifact creation).
4. Stage 1 horizontal span preservation and diacritic vertical context padding.
5. Stage 2 conversational preamble & filler peeling.
"""

import json
import os
import shutil
import tempfile
import unittest
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.evaluate_vitextvqa import (
    CheckpointManager,
    aggregate_shards,
    generate_benchmark_report,
)
from scripts.benchmark.sg_vcd.pipeline import strip_conversational_preamble
from scripts.benchmark.sg_vcd.roi_cropper import IsotropicROICropper


class TestMultiWorkerShardingEngine(unittest.TestCase):
    """Verifies deterministic sharding, per-shard I/O, and atomic aggregation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_deterministic_sharding_partitioning(self):
        """Verify partition math i % num_workers == worker_id yields complete non-overlapping partitions."""
        total_samples = 20
        samples = [{"global_index": i, "question_id": 1000 + i} for i in range(total_samples)]

        for num_workers in [2, 3, 4, 5]:
            partitioned = []
            for w_id in range(num_workers):
                shard = [s for s in samples if s["global_index"] % num_workers == w_id]
                partitioned.extend(shard)

            # Assert exact count and zero dropped/duplicated items
            self.assertEqual(len(partitioned), total_samples)
            seen_ids = [s["global_index"] for s in partitioned]
            self.assertEqual(len(seen_ids), len(set(seen_ids)))
            self.assertEqual(sorted(seen_ids), list(range(total_samples)))

    def test_02_per_shard_checkpoint_manager_isolation(self):
        """Verify CheckpointManager uses shard-specific paths when _shard_ is present."""
        w0_ckpt = os.path.join(self.test_dir, "checkpoint_shard_0.json")
        w0_jsonl = os.path.join(self.test_dir, "predictions_shard_0.jsonl")

        mgr0 = CheckpointManager(checkpoint_path=w0_ckpt, output_dir=self.test_dir, jsonl_path=w0_jsonl)
        self.assertEqual(mgr0.checkpoint_path, w0_ckpt)
        self.assertEqual(mgr0.jsonl_path, w0_jsonl)
        self.assertIn("shard_0", mgr0.summary_metrics_path)

        # Append dummy record
        from scripts.benchmark.metrics import MetricsAggregator
        agg = MetricsAggregator()
        rec = {
            "sample_key": "100_img1.jpg",
            "global_index": 0,
            "question_id": 100,
            "image_name": "img1.jpg",
            "pred_answer": "test",
            "gt_answers": ["test"],
            "verified_all_answers": ["test"],
            "stage1_metrics": {"iou": 0.8, "recall": 0.9},
            "stage2_metrics": {"em_verified": 1, "f1_verified": 1.0, "substring_verified": 1.0, "anls_verified": 1.0},
            "latency_s": 0.1,
        }
        mgr0.append_record(rec, agg, method="sg_vcd", model_name="test_model")

        self.assertTrue(os.path.isfile(w0_ckpt))
        self.assertTrue(os.path.isfile(w0_jsonl))
        self.assertFalse(os.path.isfile(os.path.join(self.test_dir, "checkpoint.json")))

    def test_03_atomic_aggregation_merges_and_sorts_all_shards(self):
        """Verify aggregate_shards creates unified master artifacts sorted by sample ID."""
        num_workers = 2
        # Create shard 0 (even indices) and shard 1 (odd indices)
        records_shard_0 = [
            {
                "sample_key": f"{i}_img.jpg",
                "global_index": i,
                "index": i,
                "question_id": 1000 + i,
                "image_name": "img.jpg",
                "question": f"Question {i}",
                "pred_answer": f"Answer {i}",
                "gt_answers": [f"Answer {i}"],
                "verified_all_answers": [f"Answer {i}"],
                "added_gt_answers": [],
                "stage1_metrics": {"iou": 0.75, "recall": 0.85},
                "stage2_metrics": {"em_verified": 1, "em_gt": 1, "f1_verified": 1.0, "f1_gt": 1.0, "substring_verified": 1.0, "anls_verified": 1.0},
                "latency_s": 0.05,
            }
            for i in range(0, 10, 2)
        ]
        records_shard_1 = [
            {
                "sample_key": f"{i}_img.jpg",
                "global_index": i,
                "index": i,
                "question_id": 1000 + i,
                "image_name": "img.jpg",
                "question": f"Question {i}",
                "pred_answer": f"Answer {i}",
                "gt_answers": [f"Answer {i}"],
                "verified_all_answers": [f"Answer {i}"],
                "added_gt_answers": [],
                "stage1_metrics": {"iou": 0.70, "recall": 0.80},
                "stage2_metrics": {"em_verified": 1, "em_gt": 1, "f1_verified": 1.0, "f1_gt": 1.0, "substring_verified": 1.0, "anls_verified": 1.0},
                "latency_s": 0.06,
            }
            for i in range(1, 10, 2)
        ]

        with open(os.path.join(self.test_dir, "predictions_shard_0.jsonl"), "w", encoding="utf-8") as f:
            for r in records_shard_0:
                f.write(json.dumps(r) + "\n")

        with open(os.path.join(self.test_dir, "predictions_shard_1.jsonl"), "w", encoding="utf-8") as f:
            for r in records_shard_1:
                f.write(json.dumps(r) + "\n")

        summary = aggregate_shards(
            output_dir=self.test_dir,
            num_workers=num_workers,
            method="sg_vcd",
            model_name="5CD-AI/Vintern-1B-v3_5",
            total_time=12.5,
        )

        self.assertEqual(summary["total_evaluated_samples"], 10)
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "predictions.jsonl")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "predictions.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "summary_metrics.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "summary_metrics.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "checkpoint.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "BENCHMARK_REPORT.md")))

        # Check sorted order in predictions.jsonl
        with open(os.path.join(self.test_dir, "predictions.jsonl"), "r", encoding="utf-8") as f:
            merged_lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(merged_lines), 10)
        indices = [r["global_index"] for r in merged_lines]
        self.assertEqual(indices, list(range(10)))


class TestStage1AndStage2Upgrades(unittest.TestCase):
    """Verifies Stage 1 vertical padding and Stage 2 conversational preamble stripping."""

    def test_01_stage1_diacritic_vertical_padding(self):
        """Verify ROICropper enforces vertical padding pad_y >= 0.15 * raw_h."""
        cropper = IsotropicROICropper(target_size=448, context_margin=0.15)
        img = Image.new("RGB", (1000, 1000), color=(128, 128, 128))
        bbox = (100, 200, 300, 300)  # w=200, h=100 -> pad_y should be >= 15
        crop_res = cropper.crop(img, bbox)

        src_x1, src_y1, src_x2, src_y2 = crop_res.source_bbox
        # Top padding: 200 - src_y1 >= 15
        self.assertGreaterEqual(200 - src_y1, 15)
        # Bottom padding: src_y2 - 300 >= 15
        self.assertGreaterEqual(src_y2 - 300, 15)

    def test_02_stage2_conversational_preamble_stripping(self):
        """Verify strip_conversational_preamble peels generalized Vietnamese conversational fillers."""
        test_cases = [
            ("Bức ảnh này là Trường Đại học Bách Khoa", "Trường Đại học Bách Khoa"),
            ("Trong ảnh có chữ: Cấm Đậu Xe", "Cấm Đậu Xe"),
            ("Câu trả lời là: 50.000 VNĐ", "50.000 VNĐ"),
            ("Trong bức ảnh có chữ 'Siêu Thị Big C'", "Siêu Thị Big C"),
            ("Hình ảnh cho thấy: Xe buýt số 150", "Xe buýt số 150"),
            ("Đáp án chính xác là: 2023", "2023"),
            ("Đây là: Quán Cà Phê Trung Nguyên", "Quán Cà Phê Trung Nguyên"),
            ("Có thể thấy rằng: Màu vàng", "Màu vàng"),
            ("Theo thông tin trong ảnh, kết quả là 123", "123"),
            ("Trả lời ngắn gọn: Viettel Telecom", "Viettel Telecom"),
            ('"Nhà thuốc Long Châu"', "Nhà thuốc Long Châu"),
        ]

        for raw_input, expected in test_cases:
            cleaned = strip_conversational_preamble(raw_input)
            self.assertEqual(cleaned, expected, f"Failed to clean '{raw_input}' -> got '{cleaned}'")


if __name__ == "__main__":
    unittest.main()
