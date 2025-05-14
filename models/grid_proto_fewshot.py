"""
ALPNet with SSFP Integration
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpmodule import MultiProtoAsConv # Giữ nguyên module này
from .backbone.torchvision_backbones import BackboneEncode
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder
# DEBUG
from pdb import set_trace

import pickle
import torchvision

# options for type of prototypes
FG_PROT_MODE = 'gridconv+' # using both local and global prototype
BG_PROT_MODE = 'gridconv'  # using local prototype only. Also 'mask' refers to using global prototype only (as done in vanilla PANet)

# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95

# Biến DEBUG_SHAPES toàn cục (bạn nên quản lý qua config)
DEBUG_SHAPES = False # Đặt thành True để bật debug

class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, backbone=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.backbone = backbone
        self.config = cfg or {'align': False}

        global DEBUG_SHAPES
        DEBUG_SHAPES = self.config.get('debug_shapes', False)
        if DEBUG_SHAPES: print(f"[DEBUG] FewShotSeg __init__: DEBUG_SHAPES set to {DEBUG_SHAPES}")

        # SSFP specific configs
        self.use_ssfp = self.config.get('use_ssfp', False)


        if DEBUG_SHAPES and self.use_ssfp:
            print(f"[DEBUG] SSFP Enabled: use_ssfp={self.use_ssfp}, "
                  f"ssfp_confidence_thresh={self.ssfp_confidence_thresh}, "
                  f"ssfp_alpha_P_star={self.ssfp_alpha_P_star}")

        self.get_encoder(in_channels)
        self.get_cls()

    def get_encoder(self, in_channels):
        if self.config['which_model'] == 'mobile':
            self.encoder = BackboneEncode(backbone=self.backbone)
            if DEBUG_SHAPES: print(f"[DEBUG] Encoder: BackboneEncode initialized.")
        elif self.config['which_model'] == 'resnet':
            self.encoder = TVDeeplabRes101Encoder(use_coco_init=True)
            print("###### Backbone resnet101: Using ms-coco initialization ######")
        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')
        # if 1 == 1:
        if self.pretrained_path:
            state_dict = torch.load(self.pretrained_path)
            self.load_state_dict(state_dict, strict=False)
            # self.load_state_dict(torch.load("/root/ducnt/fewshot_medical_segmentor/exps/myexperiments_MIDDLE_0/mySSL_train_CHAOST2_Superpix_lbgroup0_scale_MIDDLE_vfold0_CHAOST2_Superpix_sets_0_1shot/17/snapshots/epoch72000_mit_b0_0.053_0.043.pth"), strict=False)
            # self.load_state_dict(torch.load(self.pretrained_path), strict=False)
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def get_cls(self):
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsConv(proto_grid = [proto_hw, proto_hw], feature_hw =  self.config["feature_hw"])
            if DEBUG_SHAPES: print(f"[DEBUG] Classifier: MultiProtoAsConv initialized with proto_grid={proto_hw}, feature_hw={feature_hw}.")
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')

    def _generate_map_prototype(self, features, mask):
        """
        Generates a prototype using Masked Average Pooling (MAP).
        Args:
            features: Tensor of shape [B, C, H, W] or [C, H, W]
            mask: Tensor of shape [B, 1, H, W] or [1, H, W] or [B, H, W] or [H,W]
        Returns:
            prototype: Tensor of shape [B, C] or [C]
        """
        if features.ndim == 3: # C, H, W -> 1, C, H, W
            features = features.unsqueeze(0)
        if mask.ndim == 2: # H, W -> 1, 1, H, W
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3: # B, H, W -> B, 1, H, W
             mask = mask.unsqueeze(1)
        
        # Đảm bảo mask và features có cùng số batch
        if features.shape[0] != mask.shape[0] and features.shape[0] == 1:
            features = features.expand(mask.shape[0], -1, -1, -1)
        if features.shape[0] != mask.shape[0] and mask.shape[0] == 1:
            mask = mask.expand(features.shape[0], -1, -1, -1)
            
        # Masked Average Pooling
        # print(f"Debug _generate_map_prototype: features.shape {features.shape}, mask.shape {mask.shape}")
        prototype = torch.sum(features * mask, dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1e-5) # B x C
        return prototype.squeeze(0) if prototype.shape[0] == 1 else prototype


    def _safe_norm_prototype(self, proto, eps=1e-4):
        """ Normalizes a prototype vector or a batch of prototype vectors. """
        if proto is None or proto.numel() == 0:
            return proto
        norm = torch.norm(proto, p=2, dim=-1, keepdim=True)
        norm = torch.max(norm, torch.ones_like(norm) * eps)
        return proto / norm

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz = False):
        if DEBUG_SHAPES: print("\n--- Entering FewShotSeg.forward (with SSFP logic) ---")
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        if DEBUG_SHAPES: print(f"[DEBUG] n_ways: {n_ways}, n_shots: {n_shots}, n_queries: {n_queries}")

        assert n_ways == 1, "1-way setting"
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]
        if DEBUG_SHAPES:
            print(f"[DEBUG] supp_imgs[0][0].shape: {supp_imgs[0][0].shape}") # [1,3,256,256]
            print(f"[DEBUG] qry_imgs[0].shape: {qry_imgs[0].shape}") # [1,3,256,256]

        assert sup_bsize == qry_bsize == 1 # Code này chỉ xử lý B=1 cho support và query trong 1 episode

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        if DEBUG_SHAPES: print(f"[DEBUG] imgs_concat.shape: {imgs_concat.shape}") # [2,3,256,256]

        img_fts = self.encoder(imgs_concat, low_level = False)
        
        fts_size = img_fts.shape[-2:]
        if DEBUG_SHAPES: print(f"[DEBUG] img_fts.shape: {img_fts.shape}, fts_size: {fts_size}") # [2,C_feat,16,16]

        # supp_fts: [Way, Shot, Batch_sup, C_feat, H_feat, W_feat]
        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)
        # qry_fts: [N_queries, Batch_qry, C_feat, H_feat, W_feat]
        qry_fts_orig = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)
        if DEBUG_SHAPES:
            print(f"[DEBUG] supp_fts.shape: {supp_fts.shape}") # [1,1,1,C_feat,16,16]
            print(f"[DEBUG] qry_fts_orig.shape: {qry_fts_orig.shape}") # [1,1,C_feat,16,16]

        # Masks (original size)
        fore_mask_orig = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        back_mask_orig = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)
        if DEBUG_SHAPES:
            print(f"[DEBUG] fore_mask_orig.shape: {fore_mask_orig.shape}") # [1,1,1,256,256]

        # Interpolated masks for cls_unit
        res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='bilinear') for fore_mask_w in fore_mask_orig], dim=0)
        res_bg_msk = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='bilinear') for back_mask_w in back_mask_orig], dim=0)
        if DEBUG_SHAPES:
            print(f"[DEBUG] res_fg_msk.shape: {res_fg_msk.shape}") # [1,1,1,16,16]

        align_loss = 0
        outputs = []
        assign_maps_list = [] # Sửa tên để tránh nhầm lẫn
        bg_sim_maps_list = []
        fg_sim_maps_list = []


        # Vì batch size (sup_bsize, qry_bsize) trong một episode là 1, vòng lặp epi này chỉ chạy 1 lần.
        # Nếu bạn muốn xử lý batch size > 1 cho các episodes, cần điều chỉnh logic này.
        # Hiện tại, qry_fts_orig có shape [N_queries, B_qry, C, Hf, Wf]
        # Ta sẽ lấy slice theo N_queries và B_qry (đều là 1 ở đây)
        current_qry_fts = qry_fts_orig[0] # Shape [B_qry, C, Hf, Wf], B_qry=1 -> [1, C, Hf, Wf]
        
        # ---- Initial Prediction using Support Prototypes ----
        initial_scores_list = []

        if DEBUG_SHAPES: print(f"[DEBUG] Initial Pred: Calling cls_unit for BACKGROUND...")
        # cls_unit mong đợi qry: [B_eff_q, C, Hf, Wf], sup_x: [Way_eff, Shot_eff, B_eff_s, C, Hf, Wf], sup_y: [Way_eff, Shot_eff, B_eff_s, Hf, Wf]
        # Truyền qry_fts_orig trực tiếp (N_queries, B_qry, C, Hf, Wf)
        # supp_fts (Way, Shot, B_sup, C, Hf, Wf)
        # res_bg_msk (Way, Shot, B_sup, Hf, Wf)
        bg_score, _, bg_aux_attr = self.cls_unit(qry_fts_orig, supp_fts, res_bg_msk, mode=BG_PROT_MODE, thresh=BG_THRESH, isval=isval, val_wsize=val_wsize, vis_sim=show_viz)
        initial_scores_list.append(bg_score)
        if DEBUG_SHAPES: print(f"[DEBUG] Initial Pred: BG _raw_score.shape: {bg_score.shape if bg_score is not None else 'None'}") # [1,1,16,16] (B_qry, 1_score, Hf, Wf)
        
        assign_maps_list.append(bg_aux_attr['proto_assign'])
        if show_viz and bg_aux_attr.get('raw_local_sims') is not None:
            bg_sim_maps_list.append(bg_aux_attr['raw_local_sims'])

        # Foreground scores (1 way)
        # res_fg_msk có shape [Way, Shot, B_sup, Hf, Wf]. Ta chỉ có 1 way.
        current_res_fg_msk_for_way0 = res_fg_msk[0] # Shape [Shot, B_sup, Hf, Wf]
        if DEBUG_SHAPES: print(f"[DEBUG] Initial Pred: Calling cls_unit for FOREGROUND way 0...")
        # _msk.unsqueeze(0) trong code gốc là để tạo chiều Way cho cls_unit.
        # res_fg_msk đã có chiều Way rồi.
        fg_score, _, fg_aux_attr = self.cls_unit(qry_fts_orig, supp_fts, res_fg_msk,
                                                 mode=FG_PROT_MODE if F.avg_pool2d(current_res_fg_msk_for_way0, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask',
                                                 thresh=FG_THRESH, isval=isval, val_wsize=val_wsize, vis_sim=show_viz)
        initial_scores_list.append(fg_score)
        if DEBUG_SHAPES: print(f"[DEBUG] Initial Pred: FG way 0 _raw_score.shape: {fg_score.shape if fg_score is not None else 'None'}") # [1,1,16,16]
        if show_viz and fg_aux_attr.get('raw_local_sims') is not None:
            fg_sim_maps_list.append(fg_aux_attr['raw_local_sims'])

        initial_pred = torch.cat(initial_scores_list, dim=1) # [B_qry, Num_classes, Hf, Wf]
        if DEBUG_SHAPES: print(f"[DEBUG] initial_pred.shape (concatenated scores): {initial_pred.shape}") # [1,2,16,16]

        final_pred_for_loss = initial_pred # Mặc định

        # ---- QESP Logic (Query-Enriched Support Feature Map) ----
# ---- QESP Logic (Query-Enriched Support Feature Map) ----
        if self.config.get('use_qesp', False):
            if DEBUG_SHAPES: print(f"[DEBUG] --- Applying QESP (Feature Map Update) ---")
            
            # Bắt đầu với dự đoán ban đầu và đặc trưng support gốc
            current_prediction_for_qesp = initial_pred 
            # current_support_features_for_qesp giữ nguyên shape [Way, Shot, B_sup, C, Hf, Wf]
            current_support_features_for_qesp = supp_fts.clone() # Tạo bản sao để tránh thay đổi supp_fts gốc ngoài ý muốn

            # Lặp 3 lần cho QESP
            num_qesp_iterations = 3
            for i in range(num_qesp_iterations):
                if DEBUG_SHAPES: print(f"\n[DEBUG] QESP Iteration {i+1}/{num_qesp_iterations}")

                # ---- Bước 1: Tạo Prototypes từ Query dựa trên dự đoán *hiện tại* ----
                # Sử dụng current_prediction_for_qesp từ vòng lặp trước (hoặc initial_pred cho i=0)
                bg_scores_query = current_prediction_for_qesp[:, 0, :, :] # [B_qry, Hf, Wf]
                fg_scores_query = current_prediction_for_qesp[:, 1, :, :] # [B_qry, Hf, Wf]

                # --- Lọc bằng Trung vị ---
                query_fg_pixel_mask = torch.zeros_like(fg_scores_query).unsqueeze(1) # Khởi tạo mặt nạ rỗng
                if fg_scores_query.numel() > 0 :
                    fg_median_threshold = torch.median(fg_scores_query.view(fg_scores_query.shape[0], -1), dim=1).values
                    fg_threshold_query_median = fg_median_threshold.unsqueeze(-1).unsqueeze(-1)
                    query_fg_pixel_mask = (fg_scores_query >= fg_threshold_query_median).float().unsqueeze(1) # [B_qry, 1, Hf, Wf]
                    if DEBUG_SHAPES:
                        print(f"[DEBUG] QESP Iter {i+1}: fg_median_threshold.shape: {fg_median_threshold.shape}, example value: {fg_median_threshold[0].item() if fg_scores_query.numel() > 0 else 'N/A'}")
                else:
                    if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: fg_scores_query is empty.")

                query_bg_pixel_mask = torch.zeros_like(bg_scores_query).unsqueeze(1) # Khởi tạo mặt nạ rỗng
                if bg_scores_query.numel() > 0:
                    bg_median_threshold = torch.median(bg_scores_query.view(bg_scores_query.shape[0], -1), dim=1).values
                    bg_threshold_query_median = bg_median_threshold.unsqueeze(-1).unsqueeze(-1)
                    query_bg_pixel_mask = (bg_scores_query >= bg_threshold_query_median).float().unsqueeze(1) # [B_qry, 1, Hf, Wf]
                    if DEBUG_SHAPES:
                        print(f"[DEBUG] QESP Iter {i+1}: bg_median_threshold.shape: {bg_median_threshold.shape}, example value: {bg_median_threshold[0].item() if bg_scores_query.numel() > 0 else 'N/A'}")
                else:
                     if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: bg_scores_query is empty.")

                if DEBUG_SHAPES:
                    print(f"[DEBUG] QESP Iter {i+1}: query_fg_pixel_mask (median).shape: {query_fg_pixel_mask.shape}, sum: {query_fg_pixel_mask.sum()}")
                    print(f"[DEBUG] QESP Iter {i+1}: query_bg_pixel_mask (median).shape: {query_bg_pixel_mask.shape}, sum: {query_bg_pixel_mask.sum()}")

                # --- Tạo prototype query toàn cục ---
                Pq_fg_global_norm = None
                if query_fg_pixel_mask.sum() > 0:
                    # Sử dụng đặc trưng query gốc (không thay đổi qua các vòng lặp QESP)
                    pq_fg_temp = self._generate_map_prototype(current_qry_fts, query_fg_pixel_mask) 
                    Pq_fg_global_norm = self._safe_norm_prototype(pq_fg_temp)
                    if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: Pq_fg_global_norm.shape: {Pq_fg_global_norm.shape if Pq_fg_global_norm is not None else 'None'}")
                
                Pq_bg_global_norm = None
                if query_bg_pixel_mask.sum() > 0:
                    pq_bg_temp = self._generate_map_prototype(current_qry_fts, query_bg_pixel_mask)
                    Pq_bg_global_norm = self._safe_norm_prototype(pq_bg_temp)
                    if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: Pq_bg_global_norm.shape: {Pq_bg_global_norm.shape if Pq_bg_global_norm is not None else 'None'}")

                # ---- Bước 2: Chuẩn bị Đặc trưng Support *hiện tại* ----
                # Sử dụng current_support_features_for_qesp từ vòng lặp trước (hoặc supp_fts gốc cho i=0)
                # Lấy slice [0,0,0] vì Way, Shot, B_sup đều là 1
                supp_fts_slice_this_iter = current_support_features_for_qesp[0, 0, 0].clone() # [C, Hf, Wf]
                # Mặt nạ support gốc không thay đổi
                res_fg_msk_slice = res_fg_msk[0, 0, 0]     # [Hf, Wf]
                res_bg_msk_slice = res_bg_msk[0, 0, 0]     # [Hf, Wf]

                if DEBUG_SHAPES:
                    print(f"[DEBUG] QESP Iter {i+1}: supp_fts_slice_this_iter.shape: {supp_fts_slice_this_iter.shape}")
                    print(f"[DEBUG] QESP Iter {i+1}: res_fg_msk_slice.shape: {res_fg_msk_slice.shape}")

                # ---- Bước 3: Cập nhật Đặc trưng Support ----
                # Sử dụng supp_fts_slice_this_iter và các prototype query MỚI (Pq_fg/bg_global_norm)
                alpha_fg = self.config.get('qesp_alpha_update', 0.8)
                alpha_bg = self.config.get('qesp_alpha_update', 0.8)
                updated_supp_fts_slice_this_iter = supp_fts_slice_this_iter.clone() # Tạo bản sao để cập nhật

                # Cập nhật vùng FG
                if Pq_fg_global_norm is not None:
                    pq_fg_to_use = Pq_fg_global_norm[0] if Pq_fg_global_norm.ndim == 2 else Pq_fg_global_norm # Shape [C]
                    fg_mask_for_update = res_fg_msk_slice.unsqueeze(0) # [1, Hf, Wf]
                    original_fg_features_support = supp_fts_slice_this_iter * fg_mask_for_update
                    updated_fg_part = alpha_fg * original_fg_features_support + \
                                    (1.0 - alpha_fg) * pq_fg_to_use.unsqueeze(-1).unsqueeze(-1)
                    updated_supp_fts_slice_this_iter = torch.where(fg_mask_for_update.bool(), updated_fg_part, updated_supp_fts_slice_this_iter)
                    if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: FG features in support map updated.")

                # Cập nhật vùng BG
                if Pq_bg_global_norm is not None:
                    pq_bg_to_use = Pq_bg_global_norm[0] if Pq_bg_global_norm.ndim == 2 else Pq_bg_global_norm # Shape [C]
                    bg_mask_for_update = res_bg_msk_slice.unsqueeze(0) # [1, Hf, Wf]
                    original_bg_features_support = supp_fts_slice_this_iter * bg_mask_for_update
                    updated_bg_part = alpha_bg * original_bg_features_support + \
                                    (1.0 - alpha_bg) * pq_bg_to_use.unsqueeze(-1).unsqueeze(-1)
                    updated_supp_fts_slice_this_iter = torch.where(bg_mask_for_update.bool(), updated_bg_part, updated_supp_fts_slice_this_iter)
                    if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: BG features in support map updated.")

                # ---- Chuẩn bị đặc trưng support đã cập nhật cho vòng lặp tiếp theo ----
                # Đưa về shape [Way, Shot, B_sup, C, Hf, Wf]
                supp_fts_updated_for_next_iter = updated_supp_fts_slice_this_iter.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: supp_fts_updated_for_next_iter.shape: {supp_fts_updated_for_next_iter.shape}")

                # ---- Bước 4: Tính Score Lần Nữa với Đặc trưng Support Đã Cập Nhật ----
                # Sử dụng đặc trưng query gốc (current_qry_fts) và đặc trưng support MỚI (supp_fts_updated_for_next_iter)
                # Mặt nạ support gốc (res_fg_msk, res_bg_msk) vẫn dùng để xác định vùng pooling trong cls_unit
                final_scores_list_qesp = []

                if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: Recalculating scores with updated support features...")
                # Gọi cls_unit với supp_fts_updated_for_next_iter
                bg_score_qesp, _, _ = self.cls_unit(current_qry_fts, supp_fts_updated_for_next_iter, res_bg_msk, 
                                                mode=BG_PROT_MODE, thresh=BG_THRESH, 
                                                isval=isval, val_wsize=val_wsize, vis_sim=False)
                final_scores_list_qesp.append(bg_score_qesp)
                if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: New BG score shape: {bg_score_qesp.shape if bg_score_qesp is not None else 'None'}")

                current_res_fg_msk_for_way0 = res_fg_msk[0] # Vẫn dùng mặt nạ gốc để check mode
                fg_score_qesp, _, _ = self.cls_unit(current_qry_fts, supp_fts_updated_for_next_iter, res_fg_msk,
                                                mode=FG_PROT_MODE if F.avg_pool2d(current_res_fg_msk_for_way0, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask',
                                                thresh=FG_THRESH, isval=isval, val_wsize=val_wsize, vis_sim=False)
                final_scores_list_qesp.append(fg_score_qesp)
                if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: New FG score shape: {fg_score_qesp.shape if fg_score_qesp is not None else 'None'}")
                
                # ---- Lưu kết quả dự đoán và đặc trưng support cho vòng lặp tiếp theo ----
                new_prediction = torch.cat(final_scores_list_qesp, dim=1) # [B_qry, 2, Hf, Wf]
                current_prediction_for_qesp = new_prediction
                current_support_features_for_qesp = supp_fts_updated_for_next_iter # Quan trọng: cập nhật đặc trưng support

                if DEBUG_SHAPES: print(f"[DEBUG] QESP Iter {i+1}: Updated current_prediction_for_qesp shape: {current_prediction_for_qesp.shape}")

            # ---- Sau khi kết thúc vòng lặp QESP ----
            # Dự đoán cuối cùng là kết quả từ vòng lặp cuối
            final_pred_for_loss = current_prediction_for_qesp 
            if DEBUG_SHAPES: print(f"[DEBUG] QESP: final_pred_for_loss.shape (after {num_qesp_iterations} iterations): {final_pred_for_loss.shape}")
            
        else: # Nếu không dùng QESP
            final_pred_for_loss = initial_pred
            if DEBUG_SHAPES: print(f"[DEBUG] QESP Skipped. Using initial_pred.")


        outputs.append(F.interpolate(final_pred_for_loss, size=img_size, mode='bilinear'))
        if DEBUG_SHAPES: print(f"[DEBUG] outputs[-1].shape (interpolated final pred): {outputs[-1].shape}")

        # ---- Alignment Loss ----
        if self.config['align'] and self.training:
            if DEBUG_SHAPES: print(f"[DEBUG] --- Calculating align_loss_epi ---")
            # Sử dụng final_pred_for_loss (có thể đã qua SSFP) để tính alignment loss
            # qry_fts_orig[0,0] -> [C,Hf,Wf] (vì epi index vào batch dim của qry_fts_orig, mà B_qry=1)
            # supp_fts[0,0,0] -> [C,Hf,Wf] (vì epi index vào batch dim của supp_fts, mà B_sup=1)
            # fore_mask_orig[0,0,0] -> [H,W]
            current_qry_fts_for_align = qry_fts_orig[0, 0] # Lấy feature của query_idx=0, batch_idx=0
            current_pred_for_align = final_pred_for_loss[0] # Lấy prediction của query_idx=0 (B_qry=1)
            
            # Lấy cho way 0, shot 0, batch_idx=0 của support
            current_supp_fts_for_align = supp_fts[0, 0, 0] 
            current_fore_mask_for_align = fore_mask_orig[0, 0, 0]
            current_back_mask_for_align = back_mask_orig[0, 0, 0]

            if DEBUG_SHAPES:
                print(f"[DEBUG] AlignLoss input qry_fts.shape: {current_qry_fts_for_align.shape}")
                print(f"[DEBUG] AlignLoss input pred.shape: {current_pred_for_align.shape}")
                print(f"[DEBUG] AlignLoss input supp_fts.shape: {current_supp_fts_for_align.shape}")
                print(f"[DEBUG] AlignLoss input fore_mask.shape: {current_fore_mask_for_align.shape}")
            
            align_loss_epi = self.alignLoss(current_qry_fts_for_align, current_pred_for_align,
                                            current_supp_fts_for_align.unsqueeze(0).unsqueeze(0), # Thêm lại dim Way, Shot
                                            current_fore_mask_for_align.unsqueeze(0).unsqueeze(0),# Thêm lại dim Way, Shot
                                            current_back_mask_for_align.unsqueeze(0).unsqueeze(0))# Thêm lại dim Way, Shot
            align_loss += align_loss_epi
            if DEBUG_SHAPES: print(f"[DEBUG] align_loss_epi: {align_loss_epi.item() if isinstance(align_loss_epi, torch.Tensor) else align_loss_epi}")

        output_final = torch.stack(outputs, dim=1)
        if DEBUG_SHAPES: print(f"[DEBUG] output_final.shape (stacked): {output_final.shape}") # [N_queries, B_qry, Num_classes, H, W] -> [1,1,2,256,256]
        output_final = output_final.view(-1, *output_final.shape[2:])
        if DEBUG_SHAPES: print(f"[DEBUG] output_final.shape (viewed): {output_final.shape}") # [1,2,256,256]

        final_assign_maps = torch.stack(assign_maps_list, dim=1) if assign_maps_list and assign_maps_list[0] is not None else None
        final_bg_sim_maps = torch.stack(bg_sim_maps_list, dim=1) if show_viz and bg_sim_maps_list and bg_sim_maps_list[0] is not None else None
        final_fg_sim_maps = torch.stack(fg_sim_maps_list, dim=1) if show_viz and fg_sim_maps_list and fg_sim_maps_list[0] is not None else None

        if DEBUG_SHAPES:
            print(f"[DEBUG] final_assign_maps.shape: {final_assign_maps.shape if final_assign_maps is not None else 'None'}")
            print(f"[DEBUG] final_bg_sim_maps.shape: {final_bg_sim_maps.shape if final_bg_sim_maps is not None else 'None'}")
            print(f"[DEBUG] final_fg_sim_maps.shape: {final_fg_sim_maps.shape if final_fg_sim_maps is not None else 'None'}")
            print("--- Exiting FewShotSeg.forward (with SSFP logic) ---\n")
        return output_final, align_loss / sup_bsize if sup_bsize > 0 else 0, [final_bg_sim_maps, final_fg_sim_maps], final_assign_maps


    def alignLoss(self, qry_fts_single, pred_single, supp_fts_single_way_shot, fore_mask_single_way_shot, back_mask_single_way_shot):
        """
        Compute the loss for the prototype alignment branch.
        Adjusted to receive single episode's data after batch processing.
        Args:
            qry_fts_single: embedding features for a single query image [C, H', W']
            pred_single: predicted segmentation score for that query [Num_classes, H', W']
            supp_fts_single_way_shot: embedding features for support images [Way, Shot, C, H', W']
            fore_mask_single_way_shot: foreground masks for support images [Way, Shot, H, W]
            back_mask_single_way_shot: background masks for support images [Way, Shot, H, W]
        """
        if DEBUG_SHAPES: print("\n  --- Entering alignLoss (modified for single episode) ---")
        if DEBUG_SHAPES:
            print(f"  [DEBUG] alignLoss initial qry_fts_single.shape: {qry_fts_single.shape}")
            print(f"  [DEBUG] alignLoss initial pred_single.shape: {pred_single.shape}")
            print(f"  [DEBUG] alignLoss initial supp_fts_single_way_shot.shape: {supp_fts_single_way_shot.shape}")
            print(f"  [DEBUG] alignLoss initial fore_mask_single_way_shot.shape: {fore_mask_single_way_shot.shape}")
            print(f"  [DEBUG] alignLoss initial back_mask_single_way_shot.shape: {back_mask_single_way_shot.shape}")

        n_ways, n_shots = supp_fts_single_way_shot.shape[0], supp_fts_single_way_shot.shape[1]
        if DEBUG_SHAPES: print(f"  [DEBUG] alignLoss n_ways: {n_ways}, n_shots: {n_shots}")

        # pred_single is [Num_classes, H', W'] -> argmax over Num_classes dim (dim=0)
        # Result of argmax is [H', W'].
        # keepdim=True makes it [1, H', W'] (Num_classes_reduced, H', W')
        # unsqueeze(0) makes it [1, 1, H', W'] (Batch_pred_mask, C_pred_mask, Hf, Wf)
        pred_mask = pred_single.argmax(dim=0, keepdim=True).unsqueeze(0)
        if DEBUG_SHAPES: print(f"  [DEBUG] pred_mask.shape (after argmax): {pred_mask.shape}") # Expected: [1, 1, H_feat, W_feat]

        # binary_masks will be a list of [Batch_pred_mask(1), C_pred_mask(1), H_feat, W_feat] tensors
        num_total_classes = pred_single.shape[0] # Should be 1 (BG) + n_ways (FG)
        binary_masks = [pred_mask == i for i in range(num_total_classes)]
        if DEBUG_SHAPES: print(f"  [DEBUG] len(binary_masks): {len(binary_masks)}, num_total_classes: {num_total_classes}, binary_masks[0].shape: {binary_masks[0].shape if len(binary_masks)>0 else '[]'}")

        skip_ways = []

        # qry_fts_single is [C, H', W']
        # cls_unit expects sup_x: [Way_eff, Shot_eff, B_eff_s, C, Hf, Wf]
        # Here, qry_fts_single acts as "support feature" for generating "query prototypes".
        # Treat it as 1 way, 1 shot, 1 batch.
        # qry_fts_single.unsqueeze(0) -> [1, C, Hf, Wf] (Shot_eff, C, Hf, Wf)
        # qry_fts_single.unsqueeze(0).unsqueeze(0) -> [1, 1, C, Hf, Wf] (Way_eff, Shot_eff, C, Hf, Wf)
        # qry_fts_single.unsqueeze(0).unsqueeze(0).unsqueeze(2) -> [1, 1, 1, C, Hf, Wf] (Way_eff, Shot_eff, B_eff_s, C, Hf, Wf)
        reshaped_qry_fts_for_cls = qry_fts_single.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        if DEBUG_SHAPES: print(f"  [DEBUG] reshaped_qry_fts_for_cls for cls_unit.shape: {reshaped_qry_fts_for_cls.shape}") # Expected: [1,1,1,C,H',W']

        loss = []
        for way_idx in range(n_ways): # Iterates 0 to n_ways-1 (for FG ways)
            if way_idx in skip_ways:
                continue
            if DEBUG_SHAPES: print(f"    [DEBUG] alignLoss - Way: {way_idx}")
            for shot_idx in range(n_shots):
                if DEBUG_SHAPES: print(f"      [DEBUG] alignLoss - Shot: {shot_idx}")
                # img_fts_align: current support feature to segment. Shape [C, H', W']
                img_fts_align = supp_fts_single_way_shot[way_idx, shot_idx]
                # cls_unit qry (query input): [Way_eff_q, B_eff_q, C, Hf, Wf]
                # Here, img_fts_align is the query. Treat as 1 Way, 1 Batch.
                img_fts_for_cls_qry = img_fts_align.unsqueeze(0).unsqueeze(0) # [1(Way_q),1(B_q),C,H',W']
                if DEBUG_SHAPES: print(f"      [DEBUG] img_fts_for_cls_qry.shape: {img_fts_for_cls_qry.shape}")

                # Prepare sup_y for cls_unit: [Way_eff_s, Shot_eff_s, B_eff_s, Hf, Wf]
                # binary_masks[0].float() is [Batch_pred_mask(1), C_pred_mask(1), Hf, Wf]
                # After interpolate: still [1, 1, Hf, Wf]
                # .squeeze(1) removes C_pred_mask dim: [Batch_pred_mask(1), Hf, Wf]
                # .unsqueeze(0).unsqueeze(0) adds Way_eff_s=1, Shot_eff_s=1:
                # results in [Way(1), Shot(1), Batch_pred_mask(1), Hf, Wf]

                bg_mask_interpolated = F.interpolate(binary_masks[0].float(), size=img_fts_align.shape[-2:], mode='bilinear')
                # binary_masks index for FG way: way_idx + 1 (because index 0 is BG)
                fg_mask_interpolated = F.interpolate(binary_masks[way_idx + 1].float(), size=img_fts_align.shape[-2:], mode='bilinear')

                bg_mask_for_pool = bg_mask_interpolated.squeeze(1) # [Batch_pred_mask(1), Hf, Wf]
                fg_mask_for_pool = fg_mask_interpolated.squeeze(1) # [Batch_pred_mask(1), Hf, Wf]

                qry_pred_bg_msk_for_cls = bg_mask_for_pool.unsqueeze(0).unsqueeze(0) # [Way(1),Shot(1),B_pred(1),Hf,Wf]
                qry_pred_fg_msk_for_cls = fg_mask_for_pool.unsqueeze(0).unsqueeze(0) # [Way(1),Shot(1),B_pred(1),Hf,Wf]

                if DEBUG_SHAPES:
                    print(f"      [DEBUG] Corrected qry_pred_bg_msk_for_cls.shape: {qry_pred_bg_msk_for_cls.shape}") # Expected: [1,1,1,Hf,Wf]
                    print(f"      [DEBUG] Corrected qry_pred_fg_msk_for_cls.shape: {qry_pred_fg_msk_for_cls.shape}") # Expected: [1,1,1,Hf,Wf]

                scores_align = []
                if DEBUG_SHAPES: print(f"      [DEBUG] Calling cls_unit for BACKGROUND in alignLoss...")
                # qry: [1,1,C,Hf,Wf], sup_x: [1,1,1,C,Hf,Wf], sup_y: [1,1,1,Hf,Wf]
                _raw_score_bg_align, _, _ = self.cls_unit(qry=img_fts_for_cls_qry, sup_x=reshaped_qry_fts_for_cls, sup_y=qry_pred_bg_msk_for_cls, mode=BG_PROT_MODE, thresh=BG_THRESH, isval=False, val_wsize=None) # isval and val_wsize should be passed if needed
                scores_align.append(_raw_score_bg_align)
                if DEBUG_SHAPES: print(f"      [DEBUG] Align BG _raw_score_bg_align.shape: {_raw_score_bg_align.shape if _raw_score_bg_align is not None else 'None'}")


                if DEBUG_SHAPES: print(f"      [DEBUG] Calling cls_unit for FOREGROUND in alignLoss...")
                # For F.avg_pool2d(fg_mask_for_pool, 4), fg_mask_for_pool [B_pred(1), Hf, Wf] needs a channel dim
                # It should be [B_pred(1), 1, Hf, Wf]
                mode_fg_align = FG_PROT_MODE if F.avg_pool2d(fg_mask_for_pool.unsqueeze(1), 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'
                _raw_score_fg_align, _, _ = self.cls_unit(qry=img_fts_for_cls_qry, sup_x=reshaped_qry_fts_for_cls, sup_y=qry_pred_fg_msk_for_cls, mode=mode_fg_align, thresh=FG_THRESH, isval=False, val_wsize=None)
                scores_align.append(_raw_score_fg_align)
                if DEBUG_SHAPES: print(f"      [DEBUG] Align FG _raw_score_fg_align.shape: {_raw_score_fg_align.shape if _raw_score_fg_align is not None else 'None'}")


                # scores_align contains [bg_score, fg_score_for_current_way]
                # Each score is [B_eff_q(1), 1_score_channel, Hf, Wf]
                supp_pred_align = torch.cat(scores_align, dim=1) # [B_eff_q(1), 2_classes (BG, current FG), Hf, Wf]
                if DEBUG_SHAPES: print(f"      [DEBUG] supp_pred_align.shape (align loss scores concat): {supp_pred_align.shape}")
                
                # Interpolate to original support mask size
                # fore_mask_single_way_shot shape is [Way, Shot, H, W]
                orig_mask_H, orig_mask_W = fore_mask_single_way_shot.shape[-2:]
                supp_pred_align = F.interpolate(supp_pred_align, size=(orig_mask_H, orig_mask_W), mode='bilinear')
                if DEBUG_SHAPES: print(f"      [DEBUG] supp_pred_align.shape (align loss scores interpolated): {supp_pred_align.shape}") # [1,2,H,W]

                # Construct the support Ground-Truth segmentation for the current support image
                # fore_mask_single_way_shot[way_idx, shot_idx] is [H, W]
                supp_label = torch.full_like(fore_mask_single_way_shot[way_idx, shot_idx], 255,
                                             device=img_fts_align.device).long()
                # For alignment loss, the "foreground" class is always 1, and "background" is 0
                # relative to the specific support image being reconstructed.
                supp_label[fore_mask_single_way_shot[way_idx, shot_idx] == 1] = 1 # Class 1 is the FG of this support
                supp_label[back_mask_single_way_shot[way_idx, shot_idx] == 1] = 0 # Class 0 is the BG of this support
                if DEBUG_SHAPES: print(f"      [DEBUG] supp_label.shape: {supp_label.shape}, unique values: {torch.unique(supp_label)}")
                
                # Compute Loss
                # supp_pred_align is [B_eff_q(1), 2_classes, H, W]
                # supp_label is [H, W], needs unsqueeze(0) for batch dim for cross_entropy
                current_loss = F.cross_entropy(
                    supp_pred_align, supp_label.unsqueeze(0), ignore_index=255)
                
                # Normalize by n_shots and n_ways if this is part of a larger averaging scheme
                # Based on original code, it was / n_shots / n_ways
                # However, since this function now processes one (way,shot) effectively for alignment's target,
                # the normalization might be handled outside or differently.
                # For now, let's keep it as per the original structure's intent for an item's contribution.
                if n_shots > 0 and n_ways > 0:
                    current_loss = current_loss / (n_shots * n_ways)
                
                loss.append(current_loss)
                if DEBUG_SHAPES: print(f"      [DEBUG] Align current_loss: {current_loss.item()}")

        total_align_loss = torch.sum(torch.stack(loss)) if loss else torch.tensor(0.0, device=qry_fts_single.device)
        if DEBUG_SHAPES:
            print(f"  [DEBUG] Total align_loss for episode: {total_align_loss.item()}")
            print("  --- Exiting alignLoss (modified) ---")
        return total_align_loss