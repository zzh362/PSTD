"""Defines the detector network structure."""
import torch
from torch import nn
from model.network import define_halve_unit, define_detector_block


class YetAnotherDarknet(nn.modules.Module):
    """Yet another darknet, imitating darknet-53 with depth of darknet-19."""
    def __init__(self, input_channel_size, depth_factor):
        super(YetAnotherDarknet, self).__init__()
        layers = []
        # 0
        layers += [nn.Conv2d(input_channel_size, depth_factor, kernel_size=3,
                             stride=1, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(depth_factor)]
        layers += [nn.LeakyReLU(0.1)]
        # 1
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        # 2
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        # 3
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        layers += define_detector_block(depth_factor)
        # 4
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        layers += define_detector_block(depth_factor)
        # 5
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        self.model = nn.Sequential(*layers)

    def forward(self, *x):
        return self.model(x[0])


# ==== 1. 位置编码构造函数 ====
def build_2d_sincos_position_embedding(H, W, dim, device):
    """
    构造二维正弦-余弦位置编码 [1, dim, H, W]
    """
    y_embed = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, W)
    x_embed = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).repeat(H, 1)
    omega = torch.arange(dim // 4, dtype=torch.float32, device=device) / (dim // 4)
    omega = 1. / (10000 ** omega)

    out_y = torch.einsum('hw,d->hwd', y_embed, omega)
    out_x = torch.einsum('hw,d->hwd', x_embed, omega)
    pos = torch.cat([torch.sin(out_y), torch.cos(out_y),
                     torch.sin(out_x), torch.cos(out_x)], dim=-1)
    pos = pos.permute(2, 0, 1).unsqueeze(0)  # [1, dim, H, W]
    return pos


# ==== 2. 跨时帧自注意力模块 ====
class TemporalAttention(nn.Module):
    """
    简洁的跨时帧自注意力模块（含二维位置编码）
    输入:
        feat_pre: [B, C, H, W]
        feat_now: [B, C, H, W]
    输出:
        融合后的特征: [B, C, H, W]
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim

        # Q, K, V 映射
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # 输出映射
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # 简单的归一化层
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_pre, feat_now):
        B, C, H, W = feat_now.shape

        # ==== 加入二维sin-cos位置编码 ====
        pos_embed = build_2d_sincos_position_embedding(H, W, C, feat_now.device)
        feat_pre = feat_pre + pos_embed
        feat_now = feat_now + pos_embed

        # ==== 展平为token序列 ====
        q = self.q_proj(feat_now).flatten(2).transpose(1, 2)  # [B, N, C]
        k = self.k_proj(feat_pre).flatten(2).transpose(1, 2)
        v = self.v_proj(feat_pre).flatten(2).transpose(1, 2)
        N = q.size(1)

        # ==== 多头注意力 ====
        q = q.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [B, heads, N, C//heads]
        out = out.transpose(1, 2).reshape(B, N, C)

        # ==== 恢复为空间特征图 ====
        out = out.transpose(1, 2).reshape(B, C, H, W)
        out = self.out_proj(out)

        # ==== 残差连接 + 归一化 ====
        out = out + feat_now
        out = out.flatten(2).transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out


class DirectionalPointDetector(nn.modules.Module):
    """Detector for point with direction."""
    def __init__(self, input_channel_size, depth_factor, output_channel_size):
        super(DirectionalPointDetector, self).__init__()
        self.extract_feature = YetAnotherDarknet(input_channel_size,
                                                 depth_factor)
        layers = []
        layers += define_detector_block(16 * depth_factor)
        layers += define_detector_block(16 * depth_factor)
        layers += [nn.Conv2d(32 * depth_factor, output_channel_size,
                             kernel_size=1, stride=1, padding=0, bias=False)]
        self.predict = nn.Sequential(*layers)

        self.temporal_attn = TemporalAttention(dim=1024)

    def forward(self, *x):

        pre_img = x[0][:, :, :, 0:512]
        now_img = x[0][:, :, :, 512:1024]

        feature_pre = self.extract_feature(pre_img)
        feature_now = self.extract_feature(now_img)
        
        temporal_fused = self.temporal_attn(feature_now, feature_pre)
        feature = feature_now + 0.5 * temporal_fused


        prediction = self.predict(feature)

        # 4 represents that there are 4 value: confidence, shape, offset_x,
        # offset_y, whose range is between [0, 1].
        point_pred, angle_pred, type_pred = torch.split(prediction, 4, dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        type_pred = torch.sigmoid(type_pred)
        return torch.cat((point_pred, angle_pred, type_pred), dim=1)
