"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn


def clean_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return s


def choose_device(device_str: str) -> torch.device:
    s = clean_str(device_str).lower()
    if s in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1, 1)
        return x * y


class DownBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        reduction: int = 16,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.se = SEBlock(out_channels, reduction=reduction)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else None
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.se(self.bn2(self.conv2(out)))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResidualSENetEncoder(nn.Module):
    def __init__(
        self,
        layers: Sequence[int] = (2, 2, 2, 2),
        channels: Sequence[int] = (64, 128, 256, 512),
        num_channels: int = 1,
        proj_hidden_dim: int = 1024,
        emb_dim: int = 512,
        reduction: int = 16,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        block = DownBlock3D
        self.in_channels = channels[0]

        self.conv1 = nn.Conv3d(
            num_channels,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block=block,
            out_channels=channels[0],
            blocks=layers[0],
            stride=1,
            reduction=reduction,
            drop_rate=drop_rate,
        )
        self.layer2 = self._make_layer(
            block=block,
            out_channels=channels[1],
            blocks=layers[1],
            stride=2,
            reduction=reduction,
            drop_rate=drop_rate,
        )
        self.layer3 = self._make_layer(
            block=block,
            out_channels=channels[2],
            blocks=layers[2],
            stride=2,
            reduction=reduction,
            drop_rate=drop_rate,
        )
        self.layer4 = self._make_layer(
            block=block,
            out_channels=channels[3],
            blocks=layers[3],
            stride=2,
            reduction=reduction,
            drop_rate=drop_rate,
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.emb_dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else None
        self.fc = nn.Sequential(
            nn.Linear(channels[3] * block.expansion, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def _make_layer(
        self,
        block,
        out_channels: int,
        blocks: int,
        stride: int = 1,
        reduction: int = 16,
        drop_rate: float = 0.0,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = [
            block(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                reduction=reduction,
                drop_rate=drop_rate,
            )
        ]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    downsample=None,
                    reduction=reduction,
                    drop_rate=drop_rate,
                )
            )

        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        stem = self.maxpool(x)
        l1 = self.layer1(stem)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        skips = {
            "stem": stem,
            "l1": l1,
            "l2": l2,
            "l3": l3,
        }
        return l4, skips


class ResidualSENetModelV2(nn.Module):
    def __init__(
        self,
        enc_layers: Sequence[int] = (2, 2, 2, 2),
        enc_channels: Sequence[int] = (64, 128, 256, 512),
        proj_hidden_dim: int = 1024,
        emb_dim: int = 512,
        reduction: int = 16,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.encoder = ResidualSENetEncoder(
            layers=enc_layers,
            channels=enc_channels,
            num_channels=1,
            proj_hidden_dim=proj_hidden_dim,
            emb_dim=emb_dim,
            reduction=reduction,
            drop_rate=drop_rate,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        l4, _ = self.encoder.forward_features(x)
        pooled = self.encoder.global_pool(l4).flatten(1)
        if self.encoder.emb_dropout is not None:
            pooled = self.encoder.emb_dropout(pooled)
        h = self.encoder.fc(pooled)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "weights"):
            v = ckpt.get(key, None)
            if isinstance(v, dict) and len(v) > 0:
                if all(isinstance(k, str) for k in v.keys()):
                    return v

        if len(ckpt) > 0 and all(isinstance(k, str) for k in ckpt.keys()):
            if any(torch.is_tensor(v) for v in ckpt.values()):
                return ckpt

    raise ValueError(
        "Could not extract state_dict from checkpoint. "
        "Expected raw state_dict or dict with model_state_dict/state_dict."
    )


def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    sd = dict(state_dict)

    prefixes = ("module.", "model.", "net.")
    changed = True
    while changed and len(sd) > 0:
        changed = False
        for p in prefixes:
            if all(k.startswith(p) for k in sd.keys()):
                sd = {k[len(p) :]: v for k, v in sd.items()}
                changed = True

    return sd


def load_model(
    device: Optional[torch.device] = None,
) -> nn.Module:
    model_pth = Path(__file__).resolve().parents[3] / "resources" / "model.pth"

    if device is None:
        device = choose_device("auto")

    if not model_pth.exists():
        raise FileNotFoundError(f"model checkpoint not found: {model_pth}")

    model = ResidualSENetModelV2()

    ckpt = torch.load(model_pth, map_location="cpu")
    state_dict = extract_state_dict(ckpt)
    state_dict = normalize_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if len(missing) > 0:
        print(f"[WARN] missing keys ({len(missing)}): {missing[:10]}")
    if len(unexpected) > 0:
        print(f"[WARN] unexpected keys ({len(unexpected)}): {unexpected[:10]}")

    model = model.to(device)
    model.eval()
    return model
