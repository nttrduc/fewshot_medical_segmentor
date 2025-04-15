import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionProtoMatching(nn.Module):
    def __init__(self, proto_size, feature_size, num_prototypes=1):
        super(CrossAttentionProtoMatching, self).__init__()
        self.proto_size = proto_size
        self.feature_size = feature_size
        self.num_prototypes = num_prototypes

        # Định nghĩa các layers cho cross-attention
        self.query_proj = nn.Conv2d(feature_size, feature_size, kernel_size=1)
        self.key_proj = nn.Conv2d(feature_size, feature_size, kernel_size=1)
        self.value_proj = nn.Conv2d(feature_size, feature_size, kernel_size=1)
        self.attention_fc = nn.Linear(feature_size, num_prototypes)

        # Projection để tính toán score cho prototypes
        self.proto_proj = nn.Conv2d(feature_size, num_prototypes, kernel_size=1)

    def compute_cross_attention(self, query_features, support_features, support_mask):
        if query_features.dim() != 4 or support_features.dim() != 4:
            raise ValueError(f"Inputs must be 4D tensors (B, C, H, W). Got: {query_features.shape}, {support_features.shape}")

        query = self.query_proj(query_features)
        key = self.key_proj(support_features)
        value = self.value_proj(support_features)

        B, C, H, W = query.shape
        query_flat = query.flatten(2).transpose(1, 2)   # [B, HW_q, C]
        key_flat = key.flatten(2)                       # [B, C, HW_k]
        value_flat = value.flatten(2).transpose(1, 2)   # [B, HW_k, C]

        # Tính toán attention map ban đầu
        attention_map = F.softmax(torch.matmul(query_flat, key_flat), dim=-1)  # [B, HW_q, HW_k]

        # Sử dụng support_mask để điều chỉnh attention_map
        support_mask_flat = support_mask.flatten(1)  # [B, H*W]
        support_mask_flat = support_mask_flat.unsqueeze(1)  # [B, 1, H*W]
        attention_map = attention_map * support_mask_flat  # Weight attention map by mask
        attention_map = F.normalize(attention_map, p=1, dim=-1)  # Normalize after applying mask

        context = torch.matmul(attention_map, value_flat)  # [B, HW_q, C]
        context = context.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        return context, attention_map

    def compute_prototype_assignments(self, context, support_mask):
        # context: [B, C, H, W]
        B, C, H, W = context.shape
        context_flat = context.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
        
        # Masking theo support_mask để giữ lại chỉ các vùng foreground
        support_mask_flat = support_mask.flatten(1)  # [B, H*W]
        support_mask_flat = support_mask_flat.unsqueeze(2)  # [B, H*W, 1]
        context_flat = context_flat * support_mask_flat.reshape(-1, 1)  # Apply mask on context

        proto_assign = self.attention_fc(context_flat)  # [B*H*W, num_prototypes]
        proto_assign = proto_assign.view(B, H, W, self.num_prototypes).permute(0, 3, 1, 2)  # [B, P, H, W]
        
        return proto_assign

    def compute_similarity_maps(self, query_features, support_features):
        similarity_map = F.cosine_similarity(query_features, support_features, dim=1)
        return similarity_map

    def forward(self, qry, sup_x, sup_y, mode='gridconv', thresh=0.95, isval=False, val_wsize=0, vis_sim=False):
        # mapping từ alias sang biến trong module
        query_features = qry
        support_features = sup_x
        support_mask = sup_y
        
        if query_features.dim() == 5:
            query_features = query_features.squeeze(1)
        if support_features.dim() == 6:
            support_features = support_features.squeeze(1).squeeze(1)

        # Truyền support_mask vào hàm compute_cross_attention
        context, attention_map = self.compute_cross_attention(query_features, support_features, support_mask)

        # Tính toán proto_assign và similarity
        proto_assign = self.compute_prototype_assignments(context, support_mask)
        raw_local_sims = self.compute_similarity_maps(query_features, support_features)
        raw_score = self.proto_proj(context)

        aux_attr = {
            'proto_assign': proto_assign,
            'raw_local_sims': raw_local_sims
        }

        return raw_score, None, aux_attr
