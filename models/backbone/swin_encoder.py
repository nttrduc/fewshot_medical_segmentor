import torch.nn as nn
import timm

class SwinEncoder(nn.Module):
    def __init__(self, model_name='swin_large_patch4_window7_224', pretrained=True, aux_dim_keep=64):
        super().__init__()
        print("###### NETWORK: Using Swin from timm ######")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)

        self.aux_dim_keep = aux_dim_keep
        self.low_level_channels = self.backbone.feature_info[0]['num_chs']      # thường là 96
        self.high_level_channels = self.backbone.feature_info[-3]['num_chs']   # lấy từ feats[-2]

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(self.low_level_channels, aux_dim_keep, kernel_size=1),
            # nn.ReLU(inplace=True)
            # nn.Tanh()
        )

        self.high_level_proj = nn.Sequential(
            nn.Conv2d(self.high_level_channels, 256, kernel_size=1),
            # nn.ReLU(inplace=True)
            # nn.softmax(dim=1)
            # nn.Tanh()
        )

    def forward(self, x, low_level=False):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feats = self.backbone(x)
        low_feat = feats[0]        # H/4
        high_feat = feats[-3]      # H/16 thay vì H/32

        if high_feat.shape[1] != self.high_level_channels:
            high_feat = high_feat.permute(0, 3, 1, 2)
        if low_feat.shape[1] != self.low_level_channels:
            low_feat = low_feat.permute(0, 3, 1, 2)

        high_feat = self.high_level_proj(high_feat)

        if low_level:
            low_feat = self.low_level_proj(low_feat)
            return high_feat, low_feat
        else:
            return high_feat
