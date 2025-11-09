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


class TemporalAttention(nn.Module):
    """Cross-frame self-attention for temporal feature fusion."""
    def __init__(self, dim=1024, num_heads=8):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.query_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_proj   = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj   = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, feat_now, feat_pre):
        B, C, H, W = feat_now.shape

        # 1×1 conv 降维为多头形式
        q = self.query_proj(feat_now).reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = self.key_proj(feat_pre).reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = self.value_proj(feat_pre).reshape(B, self.num_heads, C // self.num_heads, H * W)

        # 注意力计算：当前帧特征对前帧特征进行查询
        attn = torch.einsum("bnch,bnck->bnhk", q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        # 跨帧信息聚合
        fused = torch.einsum("bnhk,bnck->bnch", attn, v).reshape(B, C, H, W)
        return self.out_proj(fused)

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

        self.temporal_attn = TemporalAttention(dim=1024, num_heads=8)

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
