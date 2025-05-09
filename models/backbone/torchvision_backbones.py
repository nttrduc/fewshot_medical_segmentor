"""
Backbones supported by torchvison.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
import torchvision.transforms as T
import torch

import torchvision
from torchvision import models

class BackboneEncode(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if backbone.startswith("dpn"):
            w = "imagenet+5k"
        else:
            w = "imagenet"
        backbone = "mobilenet_v2"
        print("Khoi tao khoi tao backbone: ", backbone)    
        if backbone in [ "vgg16","vgg19","densenet121", "densenet161", "xception"]:
            self.backbone = smp.Unet(
                encoder_name=backbone,          # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=w,              # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,                      # model output channels (number of classes in your dataset)
            ).encoder
        else:   
            self.backbone = smp.DeepLabV3(
                encoder_name=backbone,          # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=w,              # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,                      # model output channels (number of classes in your dataset)
            ).encoder
        self.total_params = sum(p.numel() for p in self.backbone.parameters())

    def forward(self, x_in, low_level):
        return self.backbone(x_in)[4]

