"""
U-Net Segmentation Model with ResNet-34 Encoder
Adapted for 6-channel multispectral Sentinel-2 imagery (B2, B3, B4, B8, B11, B12).
"""

from typing import Optional
import torch
import torch.nn as nn


def build_unet_model(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 6,
    classes: int = 1,
) -> nn.Module:
    """Builds a U-Net semantic segmentation model adapted for 6-band multispectral data.

    Attempts to use `segmentation_models_pytorch` with pretrained ImageNet weights.
    Falls back to a standard PyTorch U-Net if SMP is unavailable.
    """
    try:
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # Output raw logits for numerical stability with BCEWithLogits
        )
        print(f"[Model] Initialized smp.Unet (backbone: {encoder_name}, in_channels: {in_channels}, classes: {classes})")
        return model

    except ImportError:
        print("[Model] segmentation_models_pytorch not found. Instantiating native PyTorch U-Net fallback.")
        return NativeUNet(in_channels=in_channels, out_channels=classes)


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NativeUNet(nn.Module):
    """Standard U-Net architecture for multispectral segmentation."""

    def __init__(self, in_channels: int = 6, out_channels: int = 1, base_filters: int = 32):
        super().__init__()
        bf = base_filters
        self.inc = DoubleConv(in_channels, bf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf, bf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 2, bf * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 4, bf * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 8, bf * 16))

        self.up1 = nn.ConvTranspose2d(bf * 16, bf * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(bf * 16, bf * 8)

        self.up2 = nn.ConvTranspose2d(bf * 8, bf * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(bf * 8, bf * 4)

        self.up3 = nn.ConvTranspose2d(bf * 4, bf * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(bf * 4, bf * 2)

        self.up4 = nn.ConvTranspose2d(bf * 2, bf, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(bf * 2, bf)

        self.outc = nn.Conv2d(bf, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return logits
