"""Standalone ResUNet used by the released GOES--MRMS CI model.

The module is intentionally independent of the cluster training pipeline.
Layer names match the archived V3Model exactly so the released checkpoint can
be loaded with ``strict=True``.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                               bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                          bias=False),
                nn.GroupNorm(8, out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.blocks(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.block = nn.Sequential(
            ResidualBlock(out_channels + skip_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = self.reduce(x)
        return self.block(torch.cat([x, skip], dim=1))


class ConvLSTMCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 4 * channels, 3, padding=1)
        self.channels = channels
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[channels:2 * channels].fill_(1.0)

    def forward(self, x, state):
        if state is None:
            state = (torch.zeros_like(x), torch.zeros_like(x))
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


def early_fusion_features(x, include_mrms, input_mode):
    channels = 9 if include_mrms else 7
    if x.ndim != 5 or x.shape[1] != 4 or x.shape[2] != channels:
        raise RuntimeError("Expected B,4,C,H,W with the configured channels")
    raw = x.flatten(1, 2)
    if input_mode == "raw":
        return raw
    if input_mode != "raw_delta":
        raise ValueError("Unknown input mode " + input_mode)
    goes_delta = x[:, 1:, :7] - x[:, :-1, :7]
    if include_mrms:
        pair_valid = x[:, 1:, 8:9] * x[:, :-1, 8:9]
        radar_delta = (x[:, 1:, 7:8] - x[:, :-1, 7:8]) * pair_valid
        delta = torch.cat([goes_delta, radar_delta, pair_valid], dim=2)
    else:
        delta = goes_delta
    return torch.cat([raw, delta.flatten(1, 2)], dim=1)


def radar_features(x, variant):
    if x.ndim != 5 or tuple(x.shape[1:3]) != (4, 9):
        raise RuntimeError("Expected four causal GOES+MRMS frames with validity")
    raw = x.flatten(1, 2)
    valid = x[:, :, 8:9]
    radar = x[:, :, 7:8]
    extras = []
    if variant in ("threshold", "both"):
        extras.append(((radar >= 0.5).to(x.dtype) * valid).flatten(1, 2))
    if variant in ("delta", "both"):
        pair = valid[:, 1:] * valid[:, :-1]
        delta = (radar[:, 1:] - radar[:, :-1]) * pair
        extras.extend([delta.flatten(1, 2), pair.flatten(1, 2)])
    if variant not in ("none", "threshold", "delta", "both"):
        raise ValueError(variant)
    return torch.cat([raw] + extras, dim=1) if extras else raw


class V3Model(nn.Module):
    def __init__(self, architecture="resunet", include_mrms=False, width=48,
                 input_mode="raw", radar_features_mode="none"):
        super().__init__()
        if input_mode not in {"raw", "raw_delta"}:
            raise ValueError("Unsupported input mode")
        if architecture != "resunet" and input_mode != "raw":
            raise ValueError("Explicit deltas are supported only by ResUNet")
        self.architecture = architecture
        self.include_mrms = include_mrms
        self.input_mode = input_mode
        self.radar_features_mode = radar_features_mode
        if radar_features_mode != "none" and (
            architecture != "resunet" or not include_mrms or input_mode != "raw"
        ):
            raise ValueError("Radar features require raw GOES+MRMS ResUNet")
        channels = 9 if include_mrms else 7
        encoder_channels = 4 * channels if architecture == "resunet" else channels
        if input_mode == "raw_delta":
            encoder_channels += 3 * (9 if include_mrms else 7)
        encoder_channels += {"none": 0, "threshold": 4,
                             "delta": 6, "both": 10}[radar_features_mode]
        b = width
        self.enc1 = EncoderStage(encoder_channels, b, 1)
        self.enc2 = EncoderStage(b, b * 2, 2)
        self.enc3 = EncoderStage(b * 2, b * 4, 2)
        self.enc4 = EncoderStage(b * 4, b * 8, 2)
        self.bottleneck = EncoderStage(b * 8, b * 12, 2)
        if architecture == "convlstm":
            self.temporal = ConvLSTMCell(b * 12)
        self.dec4 = DecoderStage(b * 12, b * 8, b * 8)
        self.dec3 = DecoderStage(b * 8, b * 4, b * 4)
        self.dec2 = DecoderStage(b * 4, b * 2, b * 2)
        self.dec1 = DecoderStage(b * 2, b, b)
        self.head = nn.Conv2d(b, 2, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out",
                                        nonlinearity="relu")
        nn.init.constant_(self.head.bias, -4.0)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return self.bottleneck(e4), (e1, e2, e3, e4)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 4:
            raise RuntimeError("Expected B,4,C,H,W")
        if self.architecture == "resunet":
            fused = (radar_features(x, self.radar_features_mode)
                     if self.radar_features_mode != "none"
                     else early_fusion_features(x, self.include_mrms,
                                                self.input_mode))
            z, skips = self.encode(fused)
        else:
            state = None
            for frame in x.unbind(1):
                z, skips = self.encode(frame)
                state = self.temporal(z, state)
            z = state[0]
        e1, e2, e3, e4 = skips
        z = self.dec4(z, e4)
        z = self.dec3(z, e3)
        z = self.dec2(z, e2)
        z = self.dec1(z, e1)
        return self.head(z)


class ResUNet(V3Model):
    """Convenience name for the final 46-channel two-head model."""
    def __init__(self, width=48):
        super().__init__("resunet", True, width, "raw", "both")


def build_final_model(width=48):
    return ResUNet(width=width)


def build_goes_only_model(width=48):
    return V3Model("resunet", False, width, "raw", "none")

