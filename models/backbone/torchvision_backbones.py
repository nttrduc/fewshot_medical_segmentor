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



class TVDeeplabRes101Encoder(nn.Module):
    """
    FCN-Resnet101 backbone from torchvision deeplabv3
    No ASPP is used as we found emperically it hurts performance
    """
    def __init__(self, use_coco_init, aux_dim_keep = 64, use_aspp = False):
        super().__init__()
        _model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=use_coco_init, progress=True, num_classes=21, aux_loss=None)
        if use_coco_init:
            print("###### NETWORK: Using ms-coco initialization ######")
        else:
            print("###### NETWORK: Training from scratch ######")

        _model_list = list(_model.children())
        self.aux_dim_keep = aux_dim_keep
        self.backbone = _model_list[0]
        self.localconv = nn.Conv2d(2048, 256,kernel_size = 1, stride = 1, bias = False) # reduce feature map dimension
        self.asppconv = nn.Conv2d(256, 256,kernel_size = 1, bias = False)

        _aspp = _model_list[1][0]
        _conv256 = _model_list[1][1]
        self.aspp_out = nn.Sequential(*[_aspp, _conv256] )
        self.use_aspp = use_aspp

    def forward(self, x_in, low_level=False):
        """
        Args:
            low_level: whether returning aggregated low-level features in FCN
        """
        fts = self.backbone(x_in)
        if self.use_aspp:
            fts256 = self.aspp_out(fts['out'])
            high_level_fts = fts256
        else:
            fts2048 = fts['out']
            high_level_fts = self.localconv(fts2048)

        if low_level:
            low_level_fts = fts['aux'][:, : self.aux_dim_keep]
            return high_level_fts, low_level_fts
        else:
            return high_level_fts


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
        print("Total params: ", self.total_params)
    def forward(self, x_in, low_level):
            # ifnext(self.parameters()).dtype == torch.float16:
        return self.backbone(x_in)[4]

