# -*- coding: utf-8 -*-
"""
Official SWINTS (SwinTextSpotter, CVPR 2022) Model Architecture in Pure PyTorch.
Source: https://github.com/mxin262/SwinTextSpotter (projects/SWINTS/swints/)

100% exact module hierarchy and forward pass matching:
- swints.py (SWINTS, ImgFeatExtractor)
- MaskEncoding.py (PCAMaskEncoding)
- head.py (DynamicHead, RCNNHead, DynamicConv)
- rec_stage.py (REC_STAGE, DynamicConv_v2, encoder_layer, decoder_layer, decoder_layer_worelu)
- FocalTransformer.py (FocalTransformerBlock, WindowAttentionFocal)
- transformer.py (PositionalEncoding)
- roi_seq_predictors.py (SequencePredictor, BahdanauAttnDecoderRNN, Attn)
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from timm.layers import DropPath, to_2tuple, trunc_normal_

from .backbone import Conv2d, ShapeSpec, build_swint_fpn_backbone, Mlp

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class Boxes:
    """Detectron2-compatible axis-aligned bounding boxes (N, 4) in (x1, y1, x2, y2) format."""

    def __init__(self, tensor: torch.Tensor):
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, 4))
        self.tensor = tensor

    def area(self) -> torch.Tensor:
        box = self.tensor
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    def clip(self, box_size: Tuple[int, int]) -> None:
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    def __len__(self) -> int:
        return self.tensor.shape[0]


class _ROIAlignLevel(nn.Module):
    def __init__(
        self,
        output_size: Tuple[int, int],
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input_tensor: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        if rois.numel() == 0:
            return input_tensor.new_empty((0, input_tensor.shape[1]) + self.output_size)
        return roi_align(
            input_tensor,
            rois.to(dtype=input_tensor.dtype),
            self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        )


class ROIPooler(nn.Module):
    """Detectron2-compatible multi-level FPN ROIPooler using torchvision.ops.roi_align."""

    def __init__(
        self,
        output_size: Tuple[int, int],
        scales: Tuple[float, ...],
        sampling_ratio: int,
        pooler_type: str = "ROIAlignV2",
        canonical_box_size: int = 224,
        canonical_level: int = 4,
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(output_size)
        aligned = pooler_type == "ROIAlignV2"
        self.level_poolers = nn.ModuleList(
            _ROIAlignLevel(
                self.output_size,
                spatial_scale=scale,
                sampling_ratio=sampling_ratio,
                aligned=aligned,
            )
            for scale in scales
        )
        self.min_level = int(-math.log2(scales[0]))
        self.max_level = int(-math.log2(scales[-1]))
        self.canonical_level = canonical_level
        self.canonical_box_size = canonical_box_size

    def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]) -> torch.Tensor:
        if len(box_lists) == 0:
            return torch.zeros(
                (0, x[0].shape[1]) + self.output_size, device=x[0].device, dtype=x[0].dtype
            )

        pooler_fmt_boxes = torch.cat(
            [
                torch.cat(
                    [
                        torch.full_like(
                            b.tensor[:, :1], i, dtype=b.tensor.dtype, device=b.tensor.device
                        ),
                        b.tensor,
                    ],
                    dim=1,
                )
                for i, b in enumerate(box_lists)
            ],
            dim=0,
        )

        if len(self.level_poolers) == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        box_sizes = torch.sqrt(torch.cat([b.area() for b in box_lists]))
        level_assignments = torch.floor(
            self.canonical_level + torch.log2(box_sizes / self.canonical_box_size + 1e-8)
        )
        level_assignments = (
            torch.clamp(level_assignments, min=self.min_level, max=self.max_level).to(torch.int64)
            - self.min_level
        )

        num_boxes = pooler_fmt_boxes.size(0)
        num_channels = x[0].shape[1]
        output = torch.zeros(
            (num_boxes, num_channels, self.output_size[0], self.output_size[1]),
            dtype=x[0].dtype,
            device=x[0].device,
        )

        for level, pooler in enumerate(self.level_poolers):
            inds = torch.nonzero(level_assignments == level, as_tuple=True)[0]
            if inds.numel() > 0:
                pooled_lvl = pooler(x[level], pooler_fmt_boxes[inds]).to(dtype=output.dtype)
                output.index_put_((inds,), pooled_lvl)

        return output


class ImgFeatExtractor(nn.Module):
    """ImgFeatExtractor from projects/SWINTS/swints/swints.py."""

    def __init__(self, num_proposals: int = 300):
        super().__init__()
        self.num_proposals = num_proposals

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = None
        for i, f in enumerate(features):
            if i == 0:
                x = torch.mean(torch.mean(f, -1), -1)
            else:
                x_p = torch.mean(torch.mean(f, -1), -1)
                x = x + x_p
        img_feats = x.squeeze(-1).squeeze(-1).unsqueeze(1).repeat(1, self.num_proposals, 1)
        return img_feats


class PCAMaskEncoding(nn.Module):
    """PCAMaskEncoding from projects/SWINTS/swints/MaskEncoding.py."""

    def __init__(self, dim_mask: int = 60, mask_size: int = 28):
        super().__init__()
        self.agnostic = True
        self.whiten = True
        self.sigmoid = True
        self.dim_mask = dim_mask
        self.mask_size = mask_size

        self.components = nn.Parameter(
            torch.zeros(self.dim_mask, self.mask_size ** 2), requires_grad=False
        )
        self.explained_variances = nn.Parameter(
            torch.zeros(self.dim_mask), requires_grad=False
        )
        self.means = nn.Parameter(torch.zeros(self.mask_size ** 2), requires_grad=False)
        self._cached_whitened_components: Optional[torch.Tensor] = None
        self._cached_means_f32: Optional[torch.Tensor] = None

    def decoder(self, X: torch.Tensor, is_train: bool = False) -> torch.Tensor:
        X_f = X.float()
        if (
            self._cached_whitened_components is None
            or self._cached_whitened_components.device != X.device
        ):
            if self.whiten:
                self._cached_whitened_components = (
                    self.components.float()
                    * torch.sqrt(self.explained_variances.float().unsqueeze(1))
                ).detach()
            else:
                self._cached_whitened_components = self.components.float().detach()
            self._cached_means_f32 = (
                self.means.float().detach() if self.means is not None else None
            )
        X_transformed = torch.matmul(X_f, self._cached_whitened_components)
        if self._cached_means_f32 is not None:
            X_transformed = X_transformed + self._cached_means_f32
        if not is_train:
            if self.sigmoid:
                X_transformed = torch.sigmoid(X_transformed)
            else:
                X_transformed = torch.clamp(X_transformed, min=0.01, max=0.99)
        return X_transformed


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DynamicConv(nn.Module):
    """DynamicConv from projects/SWINTS/swints/head.py."""

    def __init__(
        self,
        hidden_dim: int = 256,
        dim_dynamic: int = 64,
        num_dynamic: int = 2,
        pooler_resolution: int = 7,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ELU(inplace=True)

        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, : self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params :].view(-1, self.dim_dynamic, self.hidden_dim)

        if features.dtype != param1.dtype:
            features = features.to(dtype=param1.dtype)
        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        if features.dtype != param2.dtype:
            features = features.to(dtype=param2.dtype)
        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


class RCNNHead(nn.Module):
    """RCNNHead from projects/SWINTS/swints/head.py."""

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 2,
        dim_feedforward: int = 2048,
        nhead: int = 8,
        dropout: float = 0.0,
        num_cls: int = 3,
        num_reg: int = 3,
        num_mask: int = 3,
        mask_dim: int = 60,
        scale_clamp: float = _DEFAULT_SCALE_CLAMP,
        bbox_weights: Tuple[float, float, float, float] = (2.0, 2.0, 1.0, 1.0),
    ):
        super().__init__()
        self.d_model = d_model

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(hidden_dim=d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ELU(inplace=True)

        cls_module = []
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ELU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        reg_module = []
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ELU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        mask_module = []
        for _ in range(num_mask):
            mask_module.append(nn.Linear(d_model, d_model, False))
            mask_module.append(nn.LayerNorm(d_model))
            mask_module.append(nn.ELU(inplace=True))
        self.mask_module = nn.ModuleList(mask_module)
        self.mask_logits = nn.Linear(d_model, mask_dim)

        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def apply_deltas(self, deltas: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def forward(
        self,
        features: List[torch.Tensor],
        bboxes: torch.Tensor,
        pro_features: torch.Tensor,
        pooler: ROIPooler,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N, nr_boxes = bboxes.shape[:2]

        proposal_boxes = [Boxes(bboxes[b]) for b in range(N)]
        roi_features = pooler(features, proposal_boxes)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        pro_features = (
            pro_features.view(nr_boxes, N, self.d_model)
            .permute(1, 0, 2)
            .reshape(1, N * nr_boxes, self.d_model)
        )
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        mask_feature = fc_feature.clone()

        for mask_layer in self.mask_module:
            mask_feature = mask_layer(mask_feature)
        mask_logits = self.mask_logits(mask_feature)

        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)

        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return (
            class_logits.view(N, nr_boxes, -1),
            pred_bboxes.view(N, nr_boxes, -1),
            obj_features,
            mask_logits.view(N, nr_boxes, -1),
        )


class WindowAttentionFocal(nn.Module):
    """WindowAttention from projects/SWINTS/swints/FocalTransformer.py."""

    def __init__(
        self,
        dim: int,
        expand_size: Tuple[int, int],
        shift_size: int,
        window_size: Tuple[int, int],
        focal_window: int,
        focal_level: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pool_method: str = "none",
    ):
        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.shift_size = shift_size
        self.pool_method = pool_method
        self.window_size = window_size
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        if self.expand_size[0] > 0 and self.focal_level > 0:
            self.relative_position_bias_table_to_neighbors = nn.Parameter(
                torch.zeros(
                    1,
                    num_heads,
                    window_size[0] * window_size[1],
                    (window_size[0] + 2 * expand_size[0]) * (window_size[1] + 2 * expand_size[1])
                    - window_size[0] * window_size[1],
                )
            )
            trunc_normal_(self.relative_position_bias_table_to_neighbors, std=0.02)

            offset_tl = torch.arange(0, expand_size[0])
            offset_br = torch.arange(0, expand_size[0]) + window_size[0] + expand_size[0]
            offset_row = torch.cat((offset_tl, offset_br)) * (window_size[0] + 2 * expand_size[0])

            row_ind = offset_row[:, None].expand(
                offset_row.shape[0], window_size[0] + 2 * expand_size[0]
            )
            col_ind = torch.arange(0, window_size[0] + 2 * expand_size[0])[None, :].expand(
                offset_row.shape[0], window_size[0] + 2 * expand_size[0]
            )
            row_picked = (row_ind + col_ind).flatten()

            col_tl = torch.arange(0, expand_size[1])
            col_br = torch.arange(0, expand_size[1]) + window_size[1] + expand_size[1]
            offset_col = torch.cat((col_tl, col_br))[None, :].expand(
                window_size[0], 2 * expand_size[1]
            )
            row_ind = (
                (torch.arange(0, window_size[0]) + expand_size[0])
                * (window_size[0] + 2 * expand_size[0])
            )[:, None].expand(window_size[0], 2 * expand_size[1])
            col_picked = (row_ind + offset_col).flatten()

            mask_roled = torch.cat((row_picked, col_picked)).sort()[0]
            self.register_buffer("valid_ind_rolled", mask_roled)

        if self.focal_level > 1:
            self.relative_position_bias_table_to_windows = nn.ParameterList()
            self.unfolds = nn.ModuleList()

            for k in range(self.focal_level - 1):
                stride = 2 ** k
                kernel_size = 2 * (self.focal_window // 2) + 2 ** k + (2 ** k - 1)
                self.unfolds.append(
                    nn.Unfold(
                        kernel_size=(kernel_size, kernel_size),
                        stride=stride,
                        padding=kernel_size // 2,
                    )
                )

                relative_position_bias_table_to_windows = nn.Parameter(
                    torch.zeros(
                        self.num_heads,
                        (self.window_size[0] + self.focal_window + 2 ** k - 2)
                        * (self.window_size[1] + self.focal_window + 2 ** k - 2),
                    )
                )
                trunc_normal_(relative_position_bias_table_to_windows, std=0.02)
                self.relative_position_bias_table_to_windows.append(
                    relative_position_bias_table_to_windows
                )

                relative_position_index_k = self.get_relative_position_index(
                    self.window_size, to_2tuple(self.focal_window + 2 ** k - 1)
                )
                self.register_buffer(f"relative_position_index_{k}", relative_position_index_k)

                if k > 0:
                    mask = torch.zeros(kernel_size, kernel_size)
                    mask[(2 ** k) - 1 :, (2 ** k) - 1 :] = 1
                    self.register_buffer(
                        f"valid_ind_unfold_{k}", mask.flatten().nonzero().squeeze(1)
                    )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def get_relative_position_index(self, window_size, pooled_window_size):
        coords_h_window = torch.arange(window_size[0]) - window_size[0] // 2
        coords_w_window = torch.arange(window_size[1]) - window_size[1] // 2
        coords_window = torch.stack(
            torch.meshgrid([coords_h_window, coords_w_window], indexing="ij")
        )
        coords_window = coords_window.permute(1, 2, 0).contiguous().unsqueeze(2)

        coords_h_pooled = torch.arange(pooled_window_size[0]) - pooled_window_size[0] // 2
        coords_w_pooled = torch.arange(pooled_window_size[1]) - pooled_window_size[1] // 2
        coords_pooled = torch.stack(
            torch.meshgrid([coords_h_pooled, coords_w_pooled], indexing="ij")
        )
        coords_pooled = coords_pooled.permute(1, 2, 0).contiguous()
        coords_pooled = coords_pooled.view(-1, coords_pooled.shape[-1])[None, None, :, :]

        relative_coords = coords_window - coords_pooled
        relative_coords[:, :, :, 0] += window_size[0] // 2 + pooled_window_size[0] // 2
        relative_coords[:, :, :, 1] += window_size[1] // 2 + pooled_window_size[1] // 2
        relative_coords[:, :, :, 0] *= window_size[1] + pooled_window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index.view(-1, relative_position_index.shape[-1])

    def window_partition_noreshape(self, x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
        B, H, W, C = x.shape
        pad_l = pad_t = 0
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        B, H, W, C = x.shape
        x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows

    def forward(
        self, x_all: List[torch.Tensor], mask_all: Optional[List[Optional[torch.Tensor]]] = None
    ) -> torch.Tensor:
        x = x_all[0]

        B, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, nH, nW, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_windows, k_windows, v_windows = (
            self.window_partition_noreshape(q, self.window_size),
            self.window_partition_noreshape(k, self.window_size),
            self.window_partition_noreshape(v, self.window_size),
        )

        if self.expand_size[0] > 0 and self.focal_level > 0:
            k_tl = torch.roll(k, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(1, 2))
            v_tl = torch.roll(v, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(1, 2))
            k_tr = torch.roll(k, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(1, 2))
            v_tr = torch.roll(v, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(1, 2))
            k_bl = torch.roll(k, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(1, 2))
            v_bl = torch.roll(v, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(1, 2))
            k_br = torch.roll(k, shifts=(self.expand_size[0], self.expand_size[1]), dims=(1, 2))
            v_br = torch.roll(v, shifts=(self.expand_size[0], self.expand_size[1]), dims=(1, 2))

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = (
                self.window_partition_noreshape(k_tl, self.window_size),
                self.window_partition_noreshape(k_tr, self.window_size),
                self.window_partition_noreshape(k_bl, self.window_size),
                self.window_partition_noreshape(k_br, self.window_size),
            )
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = (
                self.window_partition_noreshape(v_tl, self.window_size),
                self.window_partition_noreshape(v_tr, self.window_size),
                self.window_partition_noreshape(v_bl, self.window_size),
                self.window_partition_noreshape(v_br, self.window_size),
            )
            k_rolled = torch.cat(
                (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 3
            ).view(B, nH // self.window_size[0], nW // self.window_size[1], -1, C)
            v_rolled = torch.cat(
                (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), 3
            ).view(B, nH // self.window_size[0], nW // self.window_size[1], -1, C)

            k_rolled = k_rolled[:, :, :, self.valid_ind_rolled]
            v_rolled = v_rolled[:, :, :, self.valid_ind_rolled]
            k_rolled = k_rolled.view(
                B,
                nH // self.window_size[0],
                nW // self.window_size[1],
                -1,
                self.num_heads,
                C // self.num_heads,
            )
            v_rolled = v_rolled.view(
                B,
                nH // self.window_size[0],
                nW // self.window_size[1],
                -1,
                self.num_heads,
                C // self.num_heads,
            )
        else:
            k_rolled = None
            v_rolled = None

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for k_idx in range(self.focal_level - 1):
                stride = 2 ** k_idx
                x_window_pooled = x_all[k_idx + 1]
                nWh, nWw = x_window_pooled.shape[1:3]

                mask = x_window_pooled.new(nWh, nWw).fill_(1)
                unfolded_mask = (
                    self.unfolds[k_idx](mask.unsqueeze(0).unsqueeze(1))
                    .view(
                        1,
                        1,
                        self.unfolds[k_idx].kernel_size[0],
                        self.unfolds[k_idx].kernel_size[1],
                        -1,
                    )
                    .permute(0, 4, 2, 3, 1)
                    .contiguous()
                    .view(nWh * nWw // stride // stride, -1, 1)
                )

                if k_idx > 0:
                    valid_ind_unfold_k = getattr(self, f"valid_ind_unfold_{k_idx}")
                    unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                x_window_masks = x_window_masks.masked_fill(
                    x_window_masks == 0, float(-100.0)
                ).masked_fill(x_window_masks > 0, float(0.0))
                mask_all[k_idx + 1] = x_window_masks

                qkv_pooled = (
                    self.qkv(x_window_pooled)
                    .reshape(B, nWh, nWw, 3, C)
                    .permute(3, 0, 4, 1, 2)
                    .contiguous()
                )
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]

                k_pooled_k = (
                    self.unfolds[k_idx](k_pooled_k)
                    .view(
                        B,
                        C,
                        self.unfolds[k_idx].kernel_size[0],
                        self.unfolds[k_idx].kernel_size[1],
                        -1,
                    )
                    .permute(0, 4, 2, 3, 1)
                    .contiguous()
                    .view(
                        B, nWh * nWw // stride // stride, -1, self.num_heads, C // self.num_heads
                    )
                )
                v_pooled_k = (
                    self.unfolds[k_idx](v_pooled_k)
                    .view(
                        B,
                        C,
                        self.unfolds[k_idx].kernel_size[0],
                        self.unfolds[k_idx].kernel_size[1],
                        -1,
                    )
                    .permute(0, 4, 2, 3, 1)
                    .contiguous()
                    .view(
                        B, nWh * nWw // stride // stride, -1, self.num_heads, C // self.num_heads
                    )
                )

                if k_idx > 0:
                    valid_ind_unfold_k = getattr(self, f"valid_ind_unfold_{k_idx}")
                    k_pooled_k = k_pooled_k[:, :, valid_ind_unfold_k]
                    v_pooled_k = v_pooled_k[:, :, valid_ind_unfold_k]

                k_pooled.append(k_pooled_k)
                v_pooled.append(v_pooled_k)
            k_pooled = torch.cat(k_pooled, 2).view(
                B,
                nH // self.window_size[0],
                nW // self.window_size[1],
                -1,
                self.num_heads,
                C // self.num_heads,
            )
            v_pooled = torch.cat(v_pooled, 2).view(
                B,
                nH // self.window_size[0],
                nW // self.window_size[1],
                -1,
                self.num_heads,
                C // self.num_heads,
            )
        else:
            k_pooled = None
            v_pooled = None

        q_windows = q_windows.view(
            B,
            nH // self.window_size[0],
            nW // self.window_size[1],
            self.window_size[0] * self.window_size[1],
            self.num_heads,
            C // self.num_heads,
        )
        k_windows = k_windows.view(
            B,
            nH // self.window_size[0],
            nW // self.window_size[1],
            self.window_size[0] * self.window_size[1],
            self.num_heads,
            C // self.num_heads,
        )
        v_windows = v_windows.view(
            B,
            nH // self.window_size[0],
            nW // self.window_size[1],
            self.window_size[0] * self.window_size[1],
            self.num_heads,
            C // self.num_heads,
        )

        if k_rolled is not None:
            k_all = torch.cat((k_windows, k_rolled), 3)
            v_all = torch.cat((v_windows, v_rolled), 3)
        else:
            k_all = k_windows
            v_all = v_windows

        if k_pooled is not None:
            k_all = torch.cat((k_all, k_pooled), 3)
            v_all = torch.cat((v_all, v_pooled), 3)

        N = k_all.shape[3]
        q_windows = (
            q_windows.view(
                -1, self.window_size[0] * self.window_size[1], self.num_heads, C // self.num_heads
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k_all = (
            k_all.view(-1, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        )
        v_all = (
            v_all.view(-1, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        )

        N_window = self.window_size[0] * self.window_size[1]
        N_rolled = k_rolled.shape[3] if k_rolled is not None else 0

        q_windows = q_windows * self.scale
        attn = q_windows @ k_all.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn[:, :, :N_window, :N_window] = attn[
            :, :, :N_window, :N_window
        ] + relative_position_bias.unsqueeze(0)
        if self.expand_size[0] > 0 and self.focal_level > 0:
            attn[:, :, :N_window, N_window : (N_window + N_rolled)] = (
                attn[:, :, :N_window, N_window : (N_window + N_rolled)]
                + self.relative_position_bias_table_to_neighbors
            )

        if self.pool_method != "none" and self.focal_level > 1:
            offset = N_window + N_rolled
            for k_idx in range(self.focal_level - 1):
                relative_position_index_k = getattr(self, f"relative_position_index_{k_idx}")
                relative_position_bias_to_windows = self.relative_position_bias_table_to_windows[
                    k_idx
                ][:, relative_position_index_k.view(-1)].view(
                    -1,
                    self.window_size[0] * self.window_size[1],
                    (self.focal_window + 2 ** k_idx - 1) ** 2,
                )
                if k_idx > 0:
                    valid_ind_unfold_k = getattr(self, f"valid_ind_unfold_{k_idx}")
                    relative_position_bias_to_windows = relative_position_bias_to_windows[
                        :, :, valid_ind_unfold_k
                    ]
                attn[
                    :,
                    :,
                    :N_window,
                    offset : (offset + relative_position_bias_to_windows.shape[2]),
                ] = attn[
                    :,
                    :,
                    :N_window,
                    offset : (offset + relative_position_bias_to_windows.shape[2]),
                ] + relative_position_bias_to_windows.unsqueeze(
                    0
                )
                offset += relative_position_bias_to_windows.shape[2]

        if mask_all[0] is not None:
            nW = mask_all[0].shape[0]
            attn[:, :, :N_window, :N_window] = attn[:, :, :N_window, :N_window].view(
                attn.shape[0] // nW, nW, self.num_heads, N_window, N_window
            ) + mask_all[0].unsqueeze(1).unsqueeze(0)
            if self.expand_size[0] > 0 and self.focal_level > 0:
                attn[:, :, :N_window, N_window : (N_window + N_rolled)] = attn[
                    :, :, :N_window, N_window : (N_window + N_rolled)
                ].view(
                    attn.shape[0] // nW, nW, self.num_heads, N_window, N_rolled
                ) + mask_all[
                    0
                ][
                    :, :, self.valid_ind_rolled
                ].unsqueeze(
                    1
                ).unsqueeze(
                    0
                )

        if self.pool_method != "none" and self.focal_level > 1:
            offset = N_window + N_rolled
            for k_idx in range(self.focal_level - 1):
                nW = mask_all[k_idx + 1].shape[1]
                attn[:, :, :N_window, offset : (offset + mask_all[k_idx + 1].shape[2])] = (
                    attn[:, :, :N_window, offset : (offset + mask_all[k_idx + 1].shape[2])].view(
                        attn.shape[0] // nW,
                        nW,
                        self.num_heads,
                        N_window,
                        mask_all[k_idx + 1].shape[2],
                    )
                    + mask_all[k_idx + 1][:, :, None, None, :]
                ).view(-1, self.num_heads, N_window, mask_all[k_idx + 1].shape[2])
                offset += mask_all[k_idx + 1].shape[2]

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], N_window, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FocalTransformerBlock(nn.Module):
    """FocalTransformerBlock from projects/SWINTS/swints/FocalTransformer.py."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        expand_size: int = 0,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        pool_method: str = "none",
        focal_level: int = 1,
        focal_window: int = 1,
        use_layerscale: bool = False,
        layerscale_value: float = 1e-4,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_layerscale = use_layerscale

        if min(self.input_resolution) <= self.window_size:
            self.expand_size = 0
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.window_size_glo = self.window_size

        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level - 1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                if self.pool_method == "fc":
                    self.pool_layers.append(nn.Linear(window_size_glo * window_size_glo, 1))
                    self.pool_layers[-1].weight.data.fill_(1.0 / (window_size_glo * window_size_glo))
                    self.pool_layers[-1].bias.data.fill_(0)
                elif self.pool_method == "conv":
                    self.pool_layers.append(
                        nn.Conv2d(
                            dim,
                            dim,
                            kernel_size=window_size_glo,
                            stride=window_size_glo,
                            groups=dim,
                        )
                    )

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionFocal(
            dim,
            expand_size=to_2tuple(self.expand_size),
            shift_size=self.shift_size,
            window_size=to_2tuple(self.window_size),
            focal_window=focal_window,
            focal_level=focal_level,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            pool_method=pool_method,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        self.H = input_resolution[0]
        self.W = input_resolution[1]
        self.attn_mask = None

        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones(dim), requires_grad=True)

    def window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows_all = [shifted_x]
        x_window_masks_all = [self.attn_mask]

        if self.focal_level > 1 and self.pool_method != "none":
            for k in range(self.focal_level - 1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                pooled_h = math.ceil(Hp / self.window_size) * (2 ** k)
                pooled_w = math.ceil(Wp / self.window_size) * (2 ** k)
                H_pool = pooled_h * window_size_glo
                W_pool = pooled_w * window_size_glo

                x_level_k = shifted_x
                if Hp > H_pool:
                    trim_t = (Hp - H_pool) // 2
                    trim_b = Hp - H_pool - trim_t
                    x_level_k = x_level_k[:, trim_t:-trim_b]
                elif Hp < H_pool:
                    pad_t_k = (H_pool - Hp) // 2
                    pad_b_k = H_pool - Hp - pad_t_k
                    x_level_k = F.pad(x_level_k, (0, 0, 0, 0, pad_t_k, pad_b_k))

                if Wp > W_pool:
                    trim_l = (Wp - W_pool) // 2
                    trim_r = Wp - W_pool - trim_l
                    x_level_k = x_level_k[:, :, trim_l:-trim_r]
                elif Wp < W_pool:
                    pad_l_k = (W_pool - Wp) // 2
                    pad_r_k = W_pool - Wp - pad_l_k
                    x_level_k = F.pad(x_level_k, (0, 0, pad_l_k, pad_r_k))

                x_windows_noreshape = (
                    x_level_k.view(
                        B,
                        pooled_h,
                        window_size_glo,
                        pooled_w,
                        window_size_glo,
                        C,
                    )
                    .permute(0, 1, 3, 2, 4, 5)
                    .contiguous()
                )
                if self.pool_method == "mean":
                    x_windows_pooled = x_windows_noreshape.mean([3, 4])
                elif self.pool_method == "max":
                    x_windows_pooled = x_windows_noreshape.max(-2)[0].max(-2)[0].view(
                        B, pooled_h, pooled_w, C
                    )
                elif self.pool_method == "fc":
                    x_windows_noreshape = x_windows_noreshape.view(
                        B, pooled_h, pooled_w, window_size_glo * window_size_glo, C
                    ).transpose(3, 4)
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).flatten(-2)
                elif self.pool_method == "conv":
                    x_windows_noreshape = (
                        x_windows_noreshape.view(-1, window_size_glo, window_size_glo, C)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                    )
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).view(
                        B, pooled_h, pooled_w, C
                    )

                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [None]

        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)
        attn_windows = attn_windows[:, : self.window_size ** 2]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        x = x + self.drop_path(
            self.mlp(self.norm2(x))
            if (not self.use_layerscale)
            else (self.gamma_2 * self.mlp(self.norm2(x)))
        )
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal PositionalEncoding from projects/SWINTS/swints/transformer.py."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Attn(nn.Module):
    """Attn from projects/SWINTS/swints/roi_seq_predictors.py."""

    def __init__(self, method: str, hidden_size: int, embed_size: int, onehot_size: int):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.attn = nn.Linear(2 * self.hidden_size + onehot_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        max_len = encoder_outputs.size(0)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)
        v = self.v.to(dtype=energy.dtype).repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class BahdanauAttnDecoderRNN(nn.Module):
    """BahdanauAttnDecoderRNN from projects/SWINTS/swints/roi_seq_predictors.py."""

    def __init__(
        self,
        hidden_size: int,
        embed_size: int,
        output_size: int,
        n_layers: int = 1,
        dropout_p: float = 0.0,
        bidirectional: bool = False,
        onehot_size: Tuple[int, int] = (16, 16),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, embed_size)
        self.embedding.weight.data = torch.eye(embed_size)
        self.word_linear = nn.Linear(embed_size, hidden_size)
        self.attn = Attn("concat", hidden_size, embed_size, onehot_size[0] + onehot_size[1])
        self.rnn = nn.GRUCell(2 * hidden_size + onehot_size[0] + onehot_size[1], hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        word_input: torch.Tensor,
        last_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        word_embedded_onehot = self.embedding(word_input).view(1, word_input.size(0), -1)
        word_embedded = self.word_linear(word_embedded_onehot)
        if last_hidden.dtype != encoder_outputs.dtype:
            last_hidden = last_hidden.to(dtype=encoder_outputs.dtype)
        attn_weights = self.attn(last_hidden, encoder_outputs)
        context = attn_weights.to(dtype=encoder_outputs.dtype).bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)

        rnn_input = torch.cat((word_embedded.to(dtype=context.dtype), context), 2)
        last_hidden = last_hidden.view(last_hidden.size(0), -1)
        rnn_input = rnn_input.view(word_input.size(0), -1)
        # GRUCell runs in fp32 when weights are fp32 under autocast
        hidden = self.rnn(rnn_input.float(), last_hidden.float())
        if not self.training:
            output = F.softmax(self.out(hidden), dim=1)
        else:
            output = F.log_softmax(self.out(hidden), dim=1)
        return output, hidden, attn_weights


class SequencePredictor(nn.Module):
    """SequencePredictor from projects/SWINTS/swints/roi_seq_predictors.py."""

    def __init__(
        self,
        dim_in: int = 256,
        num_class: int = 107,
        resolution: Tuple[int, int] = (32, 32),
    ):
        super().__init__()
        self.seq_encoder = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.MAX_LENGTH = 100
        self.RESIZE_HEIGHT = resolution[0]
        self.RESIZE_WIDTH = resolution[1]
        x_onehot_size = int(self.RESIZE_WIDTH / 2)
        y_onehot_size = int(self.RESIZE_HEIGHT / 2)
        self.num_class = num_class
        self.seq_decoder = BahdanauAttnDecoderRNN(
            256,
            self.num_class,
            self.num_class,
            n_layers=1,
            dropout_p=0.1,
            onehot_size=(y_onehot_size, x_onehot_size),
        )
        self.criterion_seq_decoder = nn.NLLLoss(ignore_index=-1, reduction="none")
        self.rescale = nn.Upsample(
            size=(self.RESIZE_HEIGHT, self.RESIZE_WIDTH), mode="bilinear", align_corners=False
        )

        self.x_onehot = nn.Embedding(x_onehot_size, x_onehot_size)
        self.x_onehot.weight.data = torch.eye(x_onehot_size)
        self.y_onehot = nn.Embedding(y_onehot_size, y_onehot_size)
        self.y_onehot.weight.data = torch.eye(y_onehot_size)
        self._cached_loc_embed: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        decoder_targets: Optional[torch.Tensor] = None,
        word_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = x.device
        rescale_out = self.rescale(x)
        seq_decoder_input = self.seq_encoder(rescale_out)
        x_onehot_size = int(self.RESIZE_WIDTH / 2)
        y_onehot_size = int(self.RESIZE_HEIGHT / 2)

        B = seq_decoder_input.size(0)
        if (
            self._cached_loc_embed is None
            or self._cached_loc_embed.device != device
            or self._cached_loc_embed.dtype != seq_decoder_input.dtype
        ):
            x_t, y_t = np.meshgrid(
                np.linspace(0, x_onehot_size - 1, x_onehot_size),
                np.linspace(0, y_onehot_size - 1, y_onehot_size),
            )
            x_t = torch.as_tensor(x_t, dtype=torch.long, device=device)
            y_t = torch.as_tensor(y_t, dtype=torch.long, device=device)
            x_onehot_embedding = (
                self.x_onehot(x_t)
                .to(dtype=seq_decoder_input.dtype)
                .transpose(0, 2)
                .transpose(1, 2)
                .unsqueeze(0)
            )
            y_onehot_embedding = (
                self.y_onehot(y_t)
                .to(dtype=seq_decoder_input.dtype)
                .transpose(0, 2)
                .transpose(1, 2)
                .unsqueeze(0)
            )
            self._cached_loc_embed = torch.cat(
                [x_onehot_embedding, y_onehot_embedding], dim=1
            ).contiguous()

        loc_embed_batch = self._cached_loc_embed.expand(B, -1, -1, -1)
        seq_decoder_input_loc = torch.cat([seq_decoder_input, loc_embed_batch], 1)
        seq_decoder_input_reshape = (
            seq_decoder_input_loc.view(B, seq_decoder_input_loc.size(1), -1)
            .transpose(0, 2)
            .transpose(1, 2)
        )

        # Batched greedy decoding across all B proposals simultaneously on GPU
        decoder_input = torch.zeros((B, 1), dtype=torch.long, device=device)
        decoder_hidden = torch.zeros((B, 256), dtype=seq_decoder_input_reshape.dtype, device=device)
        decoded_words = torch.zeros((B, self.MAX_LENGTH), dtype=torch.int32, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)

        for di in range(self.MAX_LENGTH):
            # Only check active.any() periodically to avoid per-step GPU->CPU stream sync overhead
            if di >= 10 and (di % 6 == 0) and not active.any():
                break
            decoder_output, decoder_hidden, _ = self.seq_decoder(
                decoder_input, decoder_hidden, seq_decoder_input_reshape
            )
            _, topi = decoder_output.topk(1, dim=1)  # (B, 1)
            tok = topi.squeeze(1)
            decoded_words[:, di] = torch.where(
                active, tok.to(torch.int32), torch.zeros_like(tok, dtype=torch.int32)
            )
            active = active & (tok != 0)
            decoder_input = topi.detach()

        return decoded_words


class DynamicConv_v2(nn.Module):
    """DynamicConv_v2 from projects/SWINTS/swints/rec_stage.py."""

    def __init__(
        self,
        hidden_dim: int = 256,
        dim_dynamic: int = 64,
        num_dynamic: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ELU(inplace=True)

    def forward(self, pro_features: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, : self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params :].view(-1, self.dim_dynamic, self.hidden_dim)

        if features.dtype != param1.dtype:
            features = features.to(dtype=param1.dtype)
        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        if features.dtype != param2.dtype:
            features = features.to(dtype=param2.dtype)
        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        return features


def encoder_layer(in_c: int, out_c: int, k: int = 3, s: Tuple[int, int] = (2, 2), p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True),
    )


def decoder_layer(
    in_c: int,
    out_c: int,
    k: int = 3,
    s: int = 1,
    p: int = 1,
    mode: str = "nearest",
    scale_factor: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
) -> nn.Sequential:
    align_corners = None if mode == "nearest" else True
    return nn.Sequential(
        nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners),
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True),
    )


def decoder_layer_worelu(
    in_c: int,
    out_c: int,
    k: int = 3,
    s: int = 1,
    p: int = 1,
    mode: str = "nearest",
    scale_factor: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
) -> nn.Sequential:
    align_corners = None if mode == "nearest" else True
    return nn.Sequential(
        nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners),
        nn.Conv2d(in_c, in_c, k, s, p),
        nn.BatchNorm2d(in_c),
        nn.ReLU(True),
        nn.Conv2d(in_c, out_c, k, s, p),
    )


class REC_STAGE(nn.Module):
    """REC_STAGE from projects/SWINTS/swints/rec_stage.py."""

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 2,
        dim_feedforward: int = 2048,
        nhead: int = 8,
        dropout: float = 0.2,
        feat_size: Tuple[int, int] = (28, 28),
        rec_batch_size: int = 48,
        num_char: int = 107,
    ):
        super().__init__()
        self.d_model = d_model

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv_v2(hidden_dim=d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ELU(inplace=True)

        self.feat_size = feat_size
        self.rec_batch_size = rec_batch_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.TLSAM = nn.Sequential(
            FocalTransformerBlock(
                dim=256,
                input_resolution=self.feat_size,
                num_heads=8,
                window_size=7,
                expand_size=0,
                shift_size=0,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.2,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                pool_method="fc",
                focal_level=2,
                focal_window=3,
                use_layerscale=False,
                layerscale_value=1e-4,
            ),
            FocalTransformerBlock(
                dim=256,
                input_resolution=self.feat_size,
                num_heads=8,
                window_size=7,
                expand_size=0,
                shift_size=0,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.2,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                pool_method="fc",
                focal_level=2,
                focal_window=3,
                use_layerscale=False,
                layerscale_value=1e-4,
            ),
            FocalTransformerBlock(
                dim=256,
                input_resolution=self.feat_size,
                num_heads=8,
                window_size=7,
                expand_size=0,
                shift_size=0,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.2,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                pool_method="fc",
                focal_level=2,
                focal_window=3,
                use_layerscale=False,
                layerscale_value=1e-4,
            ),
        )

        self.pos_encoder = PositionalEncoding(
            self.d_model, max_len=(self.feat_size[0] // 4) * (self.feat_size[1] // 4)
        )
        num_channels = d_model
        in_channels = d_model
        mode = "nearest"
        self.k_encoder = nn.Sequential(
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
        )
        self.k_decoder_det = nn.Sequential(
            decoder_layer_worelu(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer_worelu(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, in_channels, size=(self.feat_size[0], self.feat_size[1]), mode=mode
            ),
        )
        self.k_decoder_rec = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
        )

        self.seq_decoder = SequencePredictor(dim_in=d_model, num_class=num_char)
        self.rescale = nn.Upsample(
            size=(self.feat_size[0], self.feat_size[1]), mode="bilinear", align_corners=False
        )

    def forward(
        self,
        roi_features: torch.Tensor,
        pro_features: torch.Tensor,
        gt_masks: torch.Tensor,
        N: int,
        nr_boxes: int,
    ) -> torch.Tensor:
        features = []
        k = roi_features
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        n, c, h, w = k.size()
        k = k.view(n, c, -1).permute(2, 0, 1)

        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        pro_features = pro_features.permute(1, 0, 2).reshape(1, -1, self.d_model)
        pro_features2 = self.inst_interact(pro_features, k)
        pro_features = k.permute(1, 0, 2) + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        obj_features = obj_features.permute(1, 0, 2)
        obj_features = self.pos_encoder(obj_features)
        obj_features = self.transformer_encoder(obj_features)
        obj_features = obj_features.permute(1, 2, 0)
        n, c, _ = obj_features.shape
        obj_features = obj_features.view(n, c, self.feat_size[0] // 4, self.feat_size[1] // 4)

        k = k.permute(1, 2, 0).view(n, c, self.feat_size[0] // 4, self.feat_size[1] // 4)
        k_rec = k * obj_features.sigmoid()
        k_rec = self.k_decoder_rec[0](k_rec) + features[0]

        k_det = self.k_decoder_det[0](obj_features) + features[0]
        k_rec = k_rec * k_det.sigmoid()

        k_rec = self.k_decoder_rec[1](k_rec) + roi_features
        k_det = self.k_decoder_det[1](k_det) + roi_features
        k_rec = k_rec * k_det.sigmoid()

        k_rec = self.k_decoder_det[-1](k_rec)
        k_rec = k_rec.flatten(-2, -1).permute(0, 2, 1)
        k_rec = self.TLSAM(k_rec)
        k_rec = k_rec.permute(0, 2, 1).view(n, c, self.feat_size[0], self.feat_size[1])
        gt_masks = self.rescale(gt_masks.unsqueeze(1))
        k_rec = k_rec * gt_masks
        attn_vecs = self.seq_decoder(k_rec)
        return attn_vecs


class DynamicHead(nn.Module):
    """DynamicHead from projects/SWINTS/swints/head.py."""

    def __init__(
        self,
        num_heads: int = 6,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dim_feedforward: int = 2048,
        nhead: int = 8,
        dropout: float = 0.0,
        train_num_proposal: int = 300,
        test_num_proposals: int = 100,
        mask_dim: int = 60,
        scales: Tuple[float, ...] = (0.25, 0.125, 0.0625, 0.03125),
    ):
        super().__init__()
        self.box_pooler = ROIPooler(
            output_size=(7, 7),
            scales=scales,
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
        )
        self.box_pooler_rec = ROIPooler(
            output_size=(28, 28),
            scales=scales,
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
        )

        self.hidden_dim = hidden_dim
        self.train_num_proposal = train_num_proposal
        self.test_num_proposals = test_num_proposals
        self.num_heads = num_heads
        self.mask_dim = mask_dim
        self.return_intermediate = True
        self.num_classes = num_classes

        rcnn_head = RCNNHead(
            d_model=hidden_dim,
            num_classes=num_classes,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            mask_dim=mask_dim,
        )
        self.head_series = _get_clones(rcnn_head, num_heads)

        self.rec_stage = REC_STAGE(
            d_model=hidden_dim,
            num_classes=num_classes,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
        )
        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 2, 2),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(True),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 4, 4),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(True),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(True),
                )
                for _ in range(4)
            ]
        )

    def extra_rec_feat(
        self,
        mask_encoding: PCAMaskEncoding,
        N: int,
        bboxes: torch.Tensor,
        class_logits: torch.Tensor,
        pred_bboxes: torch.Tensor,
        mask_logits: torch.Tensor,
        proposal_features: torch.Tensor,
        features: List[torch.Tensor],
    ):
        gt_masks = []
        proposal_boxes_pred = []
        pred_mask = mask_logits.detach()

        scores = torch.sigmoid(class_logits)
        labels = (
            torch.arange(2, device=bboxes.device)
            .unsqueeze(0)
            .repeat(self.train_num_proposal, 1)
            .flatten(0, 1)
        )
        inter_class_logits = []
        inter_pred_bboxes = []
        inter_pred_masks = []
        inter_pred_label = []

        for b in range(N):
            num_proposals = self.test_num_proposals
            scores_per_image, topk_indices = scores[b].flatten(0, 1).topk(
                num_proposals, sorted=False
            )
            labels_per_image = labels[topk_indices]
            box_pred_per_image = bboxes[b].view(-1, 1, 4).repeat(1, 2, 1).view(-1, 4)
            box_pred_per_image = box_pred_per_image[topk_indices]

            mask_pred_per_image = pred_mask.view(-1, self.mask_dim)
            mask_pred_per_image = mask_encoding.decoder(mask_pred_per_image, is_train=False)
            mask_pred_per_image = mask_pred_per_image.view(-1, 1, 28, 28)
            _, c, w, h = mask_pred_per_image.size()
            mask_pred_per_image = torch.repeat_interleave(mask_pred_per_image, 2, 1).view(
                -1, c, w, h
            )
            mask_pred_per_image = mask_pred_per_image[topk_indices]

            proposal_features = (
                proposal_features[b]
                .view(-1, 1, self.hidden_dim)
                .repeat(1, 2, 1)
                .view(-1, self.hidden_dim)
            )
            proposal_features = proposal_features[topk_indices]
            proposal_boxes_pred.append(Boxes(box_pred_per_image))
            gt_masks.append(mask_pred_per_image)
            inter_class_logits.append(scores_per_image)
            inter_pred_bboxes.append(box_pred_per_image)
            inter_pred_masks.append(mask_pred_per_image)
            inter_pred_label.append(labels_per_image)

        rec_map = self.box_pooler_rec(features, proposal_boxes_pred)
        gt_masks_cat = torch.cat(gt_masks)
        nr_boxes = rec_map.shape[0]
        gt_masks_b = (gt_masks_cat > 0.4).float().squeeze(1)

        return (
            inter_class_logits,
            inter_pred_bboxes,
            inter_pred_masks,
            inter_pred_label,
            proposal_features,
            gt_masks_b,
            rec_map,
            nr_boxes,
        )

    def forward(
        self,
        features: List[torch.Tensor],
        init_bboxes: torch.Tensor,
        init_features: torch.Tensor,
        mask_encoding: PCAMaskEncoding,
    ):
        bboxes = init_bboxes
        proposal_features = init_features.clone()
        for i_idx in range(len(features)):
            features[i_idx] = self.conv[i_idx](features[i_idx]) + features[i_idx]

        class_logits = None
        pred_bboxes = None
        mask_logits = None
        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features, mask_logits = rcnn_head(
                features, bboxes, proposal_features, self.box_pooler
            )
            bboxes = pred_bboxes.detach()

        N = bboxes.shape[0]
        (
            inter_class_logits,
            inter_pred_bboxes,
            inter_pred_masks,
            _,
            proposal_features,
            gt_masks,
            rec_map,
            nr_boxes,
        ) = self.extra_rec_feat(
            mask_encoding,
            N,
            bboxes,
            class_logits,
            pred_bboxes,
            mask_logits,
            proposal_features,
            features,
        )

        rec_map = self.cnn(rec_map)
        rec_proposal_features = proposal_features.clone()
        rec_result = self.rec_stage(rec_map, rec_proposal_features, gt_masks, N, nr_boxes)

        return (
            torch.stack(inter_class_logits),
            torch.stack(inter_pred_bboxes),
            torch.stack(inter_pred_masks),
            rec_result,
        )


class SWINTS(nn.Module):
    """
    Official SwinTextSpotter (SWINTS) model matching SWINTS-swin-finetune-vintext.yaml.
    """

    def __init__(
        self,
        npz_path: str,
        num_proposals: int = 300,
        test_num_proposals: int = 100,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_heads: int = 6,
        mask_dim: int = 60,
    ):
        super().__init__()
        self.in_features = ["p2", "p3", "p4", "p5"]
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.test_num_proposals = test_num_proposals
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.backbone = build_swint_fpn_backbone(
            in_features=("stage2", "stage3", "stage4", "stage5"),
            out_channels=hidden_dim,
        )
        self.size_divisibility = self.backbone.size_divisibility

        self.pos_embeddings = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)

        self.IFE = ImgFeatExtractor(num_proposals=self.num_proposals)
        self.mask_encoding = PCAMaskEncoding(dim_mask=mask_dim, mask_size=28)

        parameters = np.load(npz_path)
        self.mask_encoding.components.data.copy_(
            torch.from_numpy(parameters["components_c"][0]).float()
        )
        self.mask_encoding.explained_variances.data.copy_(
            torch.from_numpy(parameters["explained_variance_c"][0]).float()
        )
        self.mask_encoding.means.data.copy_(
            torch.from_numpy(parameters["mean_c"][0]).float()
        )

        self.head = DynamicHead(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            train_num_proposal=num_proposals,
            test_num_proposals=test_num_proposals,
            mask_dim=mask_dim,
        )

        self.register_buffer(
            "pixel_mean", torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1), persistent=False
        )

    @torch.inference_mode()
    def inference_single_image(
        self,
        img_rgb_tensor: torch.Tensor,
        orig_h: int,
        orig_w: int,
        confidence_threshold: float = 0.4,
        use_amp_bf16: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run full SwinTextSpotter forward pass on a single resized RGB tensor (3, H_in, W_in).
        Matches swints.py forward() and detectron2/modeling/postprocessing.py (detector_postprocess).
        Supports native hardware bfloat16 AMP (`use_amp_bf16=True`) for L4 / Ampere+ GPUs.
        """
        _, h_in, w_in = img_rgb_tensor.shape
        size_div = self.size_divisibility
        pad_h = (size_div - h_in % size_div) % size_div
        pad_w = (size_div - w_in % size_div) % size_div

        x = (img_rgb_tensor.unsqueeze(0) - self.pixel_mean) / self.pixel_std
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

        amp_enabled = bool(use_amp_bf16 and x.is_cuda)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=amp_enabled, cache_enabled=True
        ):
            src = self.backbone(x)
            features = [src[f] for f in self.in_features]

            images_whwh = torch.tensor(
                [[w_in, h_in, w_in, h_in]], dtype=torch.float32, device=x.device
            )
            proposal_boxes = self.init_proposal_boxes.weight.clone()
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
            proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

            img_feats = self.IFE(features)
            pos_embeddings = self.pos_embeddings.weight[None].repeat(1, 1, 1)
            proposal_feats = img_feats + pos_embeddings

            outputs_class, outputs_coord, outputs_mask, out_rec = self.head(
                features, proposal_boxes, proposal_feats, mask_encoding=self.mask_encoding
            )

        # Post-processing in float32
        scores = (
            outputs_class[-1].squeeze(0) if outputs_class[-1].dim() > 1 else outputs_class[-1]
        ).float()
        box_pred = (
            outputs_coord[-1].squeeze(0) if outputs_coord[-1].dim() > 2 else outputs_coord[-1]
        ).float()
        mask_pred = (
            outputs_mask[-1].squeeze(0).squeeze(1)
            if outputs_mask[-1].dim() > 4
            else outputs_mask[-1].squeeze(1)
        ).float()

        keep = scores >= confidence_threshold
        kept_scores = scores[keep]
        kept_boxes = Boxes(box_pred[keep].clone())
        kept_boxes.scale(float(orig_w) / float(w_in), float(orig_h) / float(h_in))
        kept_boxes.clip((orig_h, orig_w))

        kept_masks = mask_pred[keep]
        kept_recs = out_rec[keep]

        order = torch.argsort(kept_scores, descending=True)
        return {
            "boxes": kept_boxes.tensor[order].cpu(),
            "scores": kept_scores[order].cpu(),
            "masks": kept_masks[order].cpu(),
            "recs": kept_recs[order].cpu(),
        }
