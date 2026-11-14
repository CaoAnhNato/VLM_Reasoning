# -*- coding: utf-8 -*-
"""
Non-Equilibrium Variational Free Energy (NE-VFE) Inference Pipeline:
Information-Geometric Variational Free Energy Decoding (IG-VFED).

Provides end-to-end inference interface adhering to Strict A* Evaluation Protocol.
"""

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Dict, List, Optional
from PIL import Image
import torch

from src.nevfe.adapter.vintern_adapter import VinternNEVFEAdapter
from src.nevfe.core.engine import AlgorithmConfig, SPEC_REF


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
    End-to-End Information-Geometric Variational Free Energy Decoding (IG-VFED) Pipeline.
    """

    def __init__(
        self,
        model,
        tokenizer,
        transform=None,
        device: str = "cuda",
        lambda_param: float = 0.50,
        beta_param: float = 0.05,
        gamma_param: float = 0.65,
        config: Optional[AlgorithmConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.transform = transform
        self.config = (
            config
            if config is not None
            else AlgorithmConfig(
                lambda_param=lambda_param,
                beta_param=beta_param,
                gamma_param=gamma_param,
            )
        )
        self.lambda_param = self.config.lambda_param
        self.beta_param = self.config.beta_param
        self.gamma_param = self.config.gamma_param

        self.adapter = VinternNEVFEAdapter(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            device=device,
            config=self.config,
        )

    def set_ablation_switches(
        self,
        comp_a: bool = True,
        comp_b: bool = True,
        comp_c: bool = True,
    ) -> None:
        """Dynamically configure micro-ablation toggles for Factorial Interaction testing."""
        self.config = AlgorithmConfig(
            lambda_param=self.config.lambda_param,
            beta_param=self.config.beta_param,
            gamma_param=self.config.gamma_param,
            ablation_mask=(int(bool(comp_a)), int(bool(comp_b)), int(bool(comp_c))),
        )
        if hasattr(self.adapter, "set_ablation_switches"):
            self.adapter.set_ablation_switches(comp_a=comp_a, comp_b=comp_b, comp_c=comp_c)

    def extract_visual_features(self, image: Image.Image) -> torch.Tensor:
        """Extract and cache 100% native unwarped visual features for image-level reuse."""
        return self.adapter.extract_visual_features(image)

    def run(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> NEVFEResult:
        """
        Execute IG-VFED inference on (image, question).

        Returns:
            NEVFEResult with predicted answer text, full-image bbox, and spec/config diagnostics.
        """
        t0 = time.time()
        w, h = image.size

        pred_text = self.adapter.generate(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        cfg = self.adapter.engine.config
        last_bbox = getattr(self.adapter, "_last_foveal_bbox", None)
        stage1_bbox = list(last_bbox) if last_bbox is not None else [0, 0, w, h]

        return NEVFEResult(
            pred_text=pred_text,
            stage1_bbox=stage1_bbox,
            spatial_confidence=1.0,
            timing={"latency_s": elapsed},
            diagnostics={
                "method": "nevfe",
                "spec_ref": SPEC_REF,
                "config": asdict(cfg),
                "ablation_mask": list(cfg.ablation_mask),
                "lambda_param": cfg.lambda_param,
                "beta_param": cfg.beta_param,
                "gamma_param": cfg.gamma_param,
            },
        )
