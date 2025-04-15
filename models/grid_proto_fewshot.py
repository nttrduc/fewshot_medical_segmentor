"""
ALPNet
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpmodule import MultiProtoAsConv
from .cross_prototype import CrossAttentionProtoMatching
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder
from .backbone.convnext_encoder import ConvNeXtEncoder
from .backbone.swin_encoder import SwinEncoder
from .backbone.SAM2_UNet.sam_model import SAM2UNetEncoderWrapper  # Import SAM2UNetEncoderWrapper
# DEBUG

from models.backbone.swin_unet.networks.vision_transformer import SwinUnet
from models.backbone.swin_unet.config import get_config
# import torch
# from pdb import set_trace

# import pickle
# import torchvision

# options for type of prototypes
FG_PROT_MODE = 'gridconv+' # using both local and global prototype
BG_PROT_MODE = 'gridconv'  # using local prototype only. Also 'mask' refers to using global prototype only (as done in vanilla PANet)

# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95
class FewShotSeg(nn.Module):
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, eval_pyramid=False):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.get_encoder(in_channels)
        self.get_cls()
#         self.fg_thresh_param = nn.Parameter(torch.tensor(0.95).logit())  # sigmoid(0.5) = 0.5
#         self.bg_thresh_param = nn.Parameter(torch.tensor(0.95).logit())  # tùy chọn


    def get_encoder(self, in_channels):
        if self.config['which_model'] == 'swin':
            self.encoder = SwinEncoder()
        elif self.config['which_model'] == 'dlfcn_res101':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)
        elif self.config['which_model'] == 'convnext':
            self.encoder = ConvNeXtEncoder()
        elif self.config['which_model'] == 'sam2unet':
            self.encoder = SAM2UNetEncoderWrapper("/root/ducnt/fewshot_medical_segmentor/Pretrained checkpoints/SAM2UNet-Polyp.pth", 256)
        elif self.config['which_model'] == 'swin_unet':
            config = get_config()
            self.encoder = SwinUnet(config=config)
        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        if self.pretrained_path:
            
            # self.load_state_dict(torch.load(self.pretrained_path))
            
            
            # Bước 2: Load state_dict chứa key mới (ví dụ từ model đã có pyramid_fusion)
            loaded_state = torch.load(self.pretrained_path)

            # Bước 3: Lấy state_dict của model cũ để biết các key được phép
            model_state = self.state_dict()

            # Bước 4: Lọc bỏ các key không khớp với model cũ
            filtered_state = {k: v for k, v in loaded_state.items() if k in model_state}

            # Bước 5: Load state_dict đã lọc vào model cũ
            self.load_state_dict(filtered_state, strict=False)
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def get_cls(self):
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsConv(proto_grid=[proto_hw, proto_hw], feature_hw=feature_hw)
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False):

        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-shot has not been implemented yet" 
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)], dim=0)
        img_fts = self.encoder(imgs_concat, low_level=False)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)

        align_loss = 0
        outputs = []
        visualizes = []

        for epi in range(1):
            fg_masks = []
            res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='bilinear') for fore_mask_w in fore_mask], dim=0)
            res_bg_msk = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='bilinear') for back_mask_w in back_mask], dim=0)

            scores = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []

            _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode=BG_PROT_MODE, thresh=BG_THRESH, isval=isval, val_wsize=val_wsize, vis_sim=show_viz, )
            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(aux_attr['raw_local_sims'])

            for way, _msk in enumerate(res_fg_msk):
                _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0), mode=FG_PROT_MODE if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh=FG_THRESH, isval=isval, val_wsize=val_wsize, vis_sim=show_viz)
                scores.append(_raw_score)
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])

            pred = torch.cat(scores, dim=1)
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            # Dynamic Prototype Refinement - Call this function to update the prototypes
            self.dynamic_prototype_refinement(supp_fts, res_fg_msk, res_bg_msk, pred)

            # Feature Regularization - Add feature regularization loss to the total loss
            feature_reg_loss = self.feature_regularization(qry_fts, supp_fts)
            align_loss += feature_reg_loss

            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim=1)
        bg_sim_maps = torch.stack(bg_sim_maps, dim=1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim=1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps


    def dynamic_prototype_refinement(self, supp_fts, res_fg_msk, res_bg_msk, qry_pred=None):
        """
        Momentum + Uncertainty-aware prototype refinement.
        """
        with torch.no_grad():
            new_fg_prototypes = self.calculate_prototypes(supp_fts, res_fg_msk)
            new_bg_prototypes = self.calculate_prototypes(supp_fts, res_bg_msk)

            # Optional: Use uncertainty (entropy) of query predictions to weigh the update
            if qry_pred is not None:
                uncertainty = self.calculate_uncertainty(qry_pred)  # [1, 1, H, W]
                weight = 1.0 - uncertainty  # Higher certainty => stronger update
            else:
                weight = 1.0  # Default full update

            momentum = 0.9  # Momentum hyperparameter

            # Update foreground prototypes with momentum and uncertainty weighting
            if not hasattr(self, 'fg_prototypes'):
                self.fg_prototypes = new_fg_prototypes
            else:
                self.fg_prototypes = [
                    momentum * old + (1 - momentum) * weight * new
                    for old, new in zip(self.fg_prototypes, new_fg_prototypes)
                ]

            if not hasattr(self, 'bg_prototypes'):
                self.bg_prototypes = new_bg_prototypes
            else:
                self.bg_prototypes = [
                    momentum * old + (1 - momentum) * weight * new
                    for old, new in zip(self.bg_prototypes, new_bg_prototypes)
                ]

    def calculate_uncertainty(self, pred):
        """
        Calculate uncertainty map from query prediction using entropy.
        Input: pred - logits [1, C, H, W]
        Output: uncertainty map [1, 1, H, W]
        """
        prob = F.softmax(pred, dim=1)  # [1, C, H, W]
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1, keepdim=True)  # [1, 1, H, W]
        normalized_entropy = entropy / torch.log(torch.tensor(prob.shape[1], dtype=torch.float32))  # Normalize to [0,1]
        return normalized_entropy

    def calculate_prototypes(self, supp_fts, masks):
        """
        Calculate masked average features as prototypes.
        Input shape: supp_fts [n_ways, n_shots, 1, C, H, W], masks [n_ways, n_shots, 1, H, W]
        Output: list of prototype tensors per class
        """
        n_ways, n_shots, _, C, H, W = supp_fts.shape
        supp_fts = supp_fts.view(n_ways * n_shots, C, H, W)
        masks = masks.view(n_ways * n_shots, 1, H, W)

        masked_fts = supp_fts * masks  # Element-wise masking
        sum_fts = masked_fts.sum(dim=(2, 3))  # [N, C]
        sum_mask = masks.sum(dim=(2, 3)) + 1e-5  # [N, 1]
        proto = sum_fts / sum_mask  # [N, C]
        proto = proto.view(n_ways, n_shots, C).mean(dim=1)  # [n_ways, C]

        return [p for p in proto]  # list of [C] per class



    def calculate_prototypes(self, supp_fts, masks):
        """
        Calculate new prototypes based on support features and their respective masks.
        """
        # Example: Compute average feature for each prototype based on mask
        prototypes = []
        for mask in masks:
            prototype = torch.mean(supp_fts * mask.unsqueeze(1), dim=0)
            prototypes.append(prototype)
        return prototypes

    def update_prototypes(self, fg_prototypes, bg_prototypes):
        """
        Update the stored prototypes based on new calculated prototypes.
        """
        self.fg_prototypes = fg_prototypes
        self.bg_prototypes = bg_prototypes

    def feature_regularization(self, qry_fts, supp_fts):
        """
        Calculate feature regularization loss to minimize the difference between query and support features.
        """
        # Assuming L2 regularization here
        
        if qry_fts.dim() == 6:
            qry_fts = qry_fts.squeeze(0)  # Bỏ chiều đầu
        if supp_fts.dim() == 6:
            supp_fts = supp_fts.squeeze(0)
        loss = F.mse_loss(qry_fts, supp_fts)
#         loss = F.mse_loss(qry_fts, supp_fts)
        return loss



    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1).unsqueeze(0)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        skip_ways = []
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2)

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1]
                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size=img_fts.shape[-2:], mode='bilinear')
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size=img_fts.shape[-2:], mode='bilinear')
                scores = []

                _raw_score_bg, _, _ = self.cls_unit(qry=img_fts, sup_x=qry_fts, sup_y=qry_pred_bg_msk.unsqueeze(-3), mode=BG_PROT_MODE, thresh=BG_THRESH)
                scores.append(_raw_score_bg)

                _raw_score_fg, _, _ = self.cls_unit(qry=img_fts, sup_x=qry_fts, sup_y=qry_pred_fg_msk.unsqueeze(-3), mode=FG_PROT_MODE, thresh=FG_THRESH)
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                loss.append(F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)

        return torch.sum(torch.stack(loss))
