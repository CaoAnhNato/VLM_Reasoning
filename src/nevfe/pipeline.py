# -*- coding: utf-8 -*-
"""
Non-Equilibrium Variational Free Energy (NE-VFE) Inference Pipeline.

Provides end-to-end inference interface adhering to Strict A* Evaluation Protocol.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
from PIL import Image
import torch

from src.nevfe.adapter.vintern_adapter import VinternNEVFEAdapter


@dataclass
class NEVFEResult:
    """
    Standardized result structure returned by NEVFEPipeline.
    Compatible with evaluate_vitextvqa.py benchmark harness.
    """
    pred_text: str
    stage1_bbox: List[int]
    spatial_confidence: float = 1.0
    timing: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class NEVFEPipeline:
    """
    End-to-End Query-Orthogonal Variational Free Energy (QO-VFE) Inference Pipeline.
    """
    def __init__(
        self,
        model,
        tokenizer,
        transform=None,
        device: str = "cuda",
        lambda_param: float = 1.0,
        beta_param: float = 1.0,
        mu_param: float = 4.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.transform = transform
        self.lambda_param = float(lambda_param)
        self.beta_param = float(beta_param)
        self.mu_param = float(mu_param)

        self.adapter = VinternNEVFEAdapter(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            device=device,
            lambda_param=self.lambda_param,
            beta_param=self.beta_param,
            mu_param=self.mu_param,
        )

    def extract_visual_features(self, image: Image.Image) -> torch.Tensor:
        """Extract and cache visual features for image-level reuse."""
        return self.adapter.extract_vit_embeds(image)

    def run(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> NEVFEResult:
        """
        Execute NE-VFE inference on (image, question).

        Returns:
            NEVFEResult with predicted answer text and Chebyshev crop bounding box.
        """
        t0 = time.time()
        w, h = image.size

        pred_text = self.adapter.generate(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0

        crop_box = getattr(self.adapter, "_last_crop_box", None)
        stage1_bbox = list(crop_box) if crop_box is not None else [0, 0, w, h]

        return NEVFEResult(
            pred_text=pred_text,
            stage1_bbox=stage1_bbox,
            spatial_confidence=1.0,
            timing={"latency_s": elapsed},
            diagnostics={
                "method": "nevfe",
                "lambda_param": self.lambda_param,
                "beta_param": self.beta_param,
                "mu_param": self.mu_param,
            },
        )
