# -*- coding: utf-8 -*-
"""
Non-Equilibrium Variational Free Energy (NE-VFE) Inference Pipeline.

Provides end-to-end inference and shared AlgorithmConfig / ablation-mask resolution
for both Stage A probes and Stage B benchmarks.
"""

from dataclasses import asdict, dataclass, field, fields, replace
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from PIL import Image
import torch

from src.nevfe.adapter.vintern_adapter import VinternNEVFEAdapter
from src.nevfe.core.engine import AlgorithmConfig, COMPONENT_NAMES, SPEC_REF


def _trim_ws(text: str) -> str:
    chars = str(text)
    lo = 0
    hi = len(chars)
    while lo < hi and chars[lo].isspace():
        lo += 1
    while hi > lo and chars[hi - 1].isspace():
        hi -= 1
    return chars[lo:hi]


def parse_ablation_mask(
    mask_str: str,
    expected_len: Optional[int] = None,
) -> Tuple[int, ...]:
    """
    Parse compact binary ablation bitstrings ('base', '', '0', '1', '01', '10', '11').
    Rejects comma-delimited single masks like '1,0' to avoid collision with --ablation-masks.
    """
    cleaned = _trim_ws(mask_str)
    if "," in cleaned:
        raise ValueError(
            f"Comma-delimited single ablation mask is forbidden ({mask_str!r}); use compact bitstrings such as 'base', '0', '1', '01', '11'."
        )
    if cleaned == "" or cleaned.lower() == "base":
        bits: Tuple[int, ...] = ()
    elif all(ch in ("0", "1") for ch in cleaned):
        bits = tuple(int(ch) for ch in cleaned)
    else:
        raise ValueError(f"Invalid compact ablation mask: {mask_str!r}")

    if expected_len is not None and len(bits) != expected_len:
        raise ValueError(
            f"Expected ablation mask length {expected_len}, got {len(bits)} from {mask_str!r}"
        )
    return bits


def resolve_algorithm_config(
    config_json: Optional[Union[str, Mapping[str, Any]]] = None,
    ablation_mask_str: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> AlgorithmConfig:
    """
    Dynamically resolve AlgorithmConfig from JSON path/mapping, scalar overrides,
    and compact ablation bitstring while preserving custom Track fields and rejecting unknown keys.
    """
    valid_fields = {f.name: f for f in fields(AlgorithmConfig)}
    kwargs: Dict[str, Any] = {}

    if config_json is not None:
        if isinstance(config_json, Mapping):
            raw_cfg = dict(config_json)
        elif isinstance(config_json, str):
            trimmed = _trim_ws(config_json)
            if trimmed:
                if os.path.isfile(trimmed):
                    with open(trimmed, "r", encoding="utf-8") as cf:
                        raw_cfg = json.load(cf)
                elif trimmed.startswith("{"):
                    raw_cfg = json.loads(trimmed)
                else:
                    raise FileNotFoundError(f"Config JSON file not found: {trimmed}")
            else:
                raw_cfg = {}
        else:
            raise TypeError(f"Unsupported config_json type: {type(config_json).__name__}")

        if not isinstance(raw_cfg, Mapping):
            raise ValueError("config_json payload must decode to a JSON object")
        unknown = sorted(set(raw_cfg.keys()) - set(valid_fields.keys()))
        if unknown:
            raise ValueError(f"Unknown AlgorithmConfig field(s) in config_json: {unknown}")
        for key, val in raw_cfg.items():
            if val is None:
                continue
            if key == "ablation_mask":
                if isinstance(val, str):
                    kwargs[key] = parse_ablation_mask(val, expected_len=len(COMPONENT_NAMES))
                elif isinstance(val, (list, tuple)):
                    kwargs[key] = tuple(int(x) for x in val)
                else:
                    raise ValueError(f"Invalid ablation_mask type in config_json: {type(val).__name__}")
            else:
                kwargs[key] = val

    if overrides:
        unknown_ov = sorted(set(overrides.keys()) - set(valid_fields.keys()))
        if unknown_ov:
            raise ValueError(f"Unknown AlgorithmConfig override key(s): {unknown_ov}")
        for key, val in overrides.items():
            if val is not None:
                kwargs[key] = val

    if ablation_mask_str is not None:
        kwargs["ablation_mask"] = parse_ablation_mask(
            ablation_mask_str,
            expected_len=len(COMPONENT_NAMES),
        )

    return AlgorithmConfig(**kwargs)


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
    End-to-End Frozen FRSA-PBCF + IG-VFED Inference Pipeline.
    """

    def __init__(
        self,
        model,
        tokenizer,
        transform=None,
        device: str = "cuda",
        lambda_param: float = 0.50,
        beta_param: float = 0.05,
        ablation_mask: Tuple[int, ...] = (),
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
                ablation_mask=ablation_mask,
            )
        )
        if abs(float(self.config.lambda_param) - 0.50) > 1e-12:
            raise ValueError(f"lambda_param must remain locked at 0.50, got {self.config.lambda_param}")
        if abs(float(self.config.beta_param) - 0.05) > 1e-12:
            raise ValueError(f"beta_param must remain locked at 0.05, got {self.config.beta_param}")
        self.lambda_param = self.config.lambda_param
        self.beta_param = self.config.beta_param

        self.adapter = VinternNEVFEAdapter(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            device=device,
            config=self.config,
        )

    def set_ablation_switches(
        self,
        *bits: Any,
        ablation_mask: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Dynamically configure variable-length ablation toggles via dataclasses.replace."""
        raw = ablation_mask if ablation_mask is not None else bits
        new_mask = tuple(int(bool(x)) for x in raw)
        self.config = replace(self.config, ablation_mask=new_mask)
        if hasattr(self.adapter, "set_ablation_switches"):
            self.adapter.set_ablation_switches(ablation_mask=new_mask)

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
        del kwargs
        t0 = time.time()
        w, h = image.size

        pred_text = self.adapter.generate(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        cfg = self.adapter.engine.config

        return NEVFEResult(
            pred_text=pred_text,
            stage1_bbox=[0, 0, w, h],
            spatial_confidence=1.0,
            timing={"latency_s": elapsed},
            diagnostics={
                "method": "nevfe",
                "spec_ref": SPEC_REF,
                "component_names": list(COMPONENT_NAMES),
                "config": asdict(cfg),
                "ablation_mask": list(cfg.ablation_mask),
                "lambda_param": cfg.lambda_param,
                "beta_param": cfg.beta_param,
                "decoder_forward_calls": int(getattr(self.adapter, "last_decoder_forward_calls", 0)),
            },
        )
