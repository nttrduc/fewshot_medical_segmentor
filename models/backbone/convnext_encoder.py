import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtEncoder(nn.Module):
    def __init__(self, use_coco_init=True, aux_dim_keep=64):
        super().__init__()
        if use_coco_init:
            print("###### NETWORK: Using ImageNet pretrained ConvNeXt ######")
            backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            print("###### NETWORK: Training ConvNeXt from scratch ######")
            backbone = convnext_tiny(weights=None)

        self.stem = backbone.features[0]
        self.stages = nn.Sequential(*backbone.features[1:])
        self.aux_dim_keep = aux_dim_keep

        self.low_level_idx = 0  # Use stem output
        self.low_level_channels = 96
        self.high_level_channels = 768

        self.low_level_proj = nn.Conv2d(self.low_level_channels, aux_dim_keep, kernel_size=1)
        self.high_level_proj = nn.Conv2d(self.high_level_channels, 256, kernel_size=1)

    def forward(self, x_in, low_level=True):
        if x_in.shape[1] == 1:
            x_in = x_in.repeat(1, 3, 1, 1)

        input_size = x_in.shape[-2:]  # e.g., 256x256
        x = self.stem(x_in)
        low_feat = x if self.low_level_idx == 0 else None

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == self.low_level_idx and self.low_level_idx != 0:
                low_feat = x

        # High-level output (resize to match ResNet's 28x28)
        high_feat = self.high_level_proj(x)
        high_feat = nn.functional.interpolate(high_feat, size=(28, 28), mode='bilinear', align_corners=False)

        if low_level:
            low_feat = self.low_level_proj(low_feat)
            low_feat = nn.functional.interpolate(low_feat, size=(28, 28), mode='bilinear', align_corners=False)
            return high_feat, low_feat
        else:
            return high_feat
