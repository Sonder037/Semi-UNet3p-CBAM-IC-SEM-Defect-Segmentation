"""
Semi-UNet3+-CBAM Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Contains both channel and spatial attention mechanisms
    """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class DownBlock(nn.Module):
    """Downsampling block"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        residual = self.conv(x)
        pooled = self.pool(residual)
        return residual, pooled


class UpBlock(nn.Module):
    """Upsampling block with skip connection"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SemiUNet3Plus_CBAM(nn.Module):
    """
    Semi-UNet3+ with CBAM implementation
    """
    def __init__(self, n_channels=3, n_classes=1, feature_scale=1, dropout=0.5):
        super(SemiUNet3Plus_CBAM, self).__init__()
        self.feature_scale = feature_scale
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Encoder
        self.down1 = DownBlock(n_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])
        self.down5 = DownBlock(filters[3], filters[4])

        # Fully connected pathway
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(filters[:-1]), filters[3], 3, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up1 = UpBlock(filters[4], filters[3])
        self.up2 = UpBlock(filters[3], filters[2])
        self.up3 = UpBlock(filters[2], filters[1])
        self.up4 = UpBlock(filters[1], filters[0])

        # Output layers
        self.out_conv = nn.Conv2d(filters[0], n_classes, 1)

        # CBAM modules
        self.cbam1 = CBAM(filters[0])
        self.cbam2 = CBAM(filters[1])
        self.cbam3 = CBAM(filters[2])
        self.cbam4 = CBAM(filters[3])
        self.cbam5 = CBAM(filters[4])

        # Upsample and downsample for feature fusion
        self.h_feature_up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.h1_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.h2_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.h3_up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.h1_down = nn.MaxPool2d(2)
        self.h2_down = nn.MaxPool2d(4)
        self.h3_down = nn.MaxPool2d(8)
        self.h4_down = nn.MaxPool2d(16)

        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        # Encoding
        h1, p1 = self.down1(x)
        h2, p2 = self.down2(p1)
        h3, p3 = self.down3(p2)
        h4, p4 = self.down4(p3)
        h5, _ = self.down5(p4)

        # Apply CBAM to encoder features
        h1 = self.cbam1(h1)
        h2 = self.cbam2(h2)
        h3 = self.cbam3(h3)
        h4 = self.cbam4(h4)
        h5 = self.cbam5(h5)

        # Full-scale skip connections
        h1_feat = self.h1_up(h1)
        h2_feat = self.h2_up(h2)
        h3_feat = self.h3_up(h3)
        h4_feat = h4
        h5_feat = self.h_feature_up(h5)

        # Fusion of all features
        fused_features = self.fusion_conv(torch.cat([h1_feat, h2_feat, h3_feat, h4_feat], dim=1))

        # Decoding
        d4 = self.up1(fused_features, h5)
        d3 = self.up2(d4, h4)
        d2 = self.up3(d3, h3)
        d1 = self.up4(d2, h2)

        # Final output
        out = self.out_conv(d1)

        if self.dropout:
            out = self.dropout(out)

        return torch.sigmoid(out)


def get_model(model_name='semiunet3plus_cbam', **kwargs):
    """
    Helper function to get model by name
    """
    if model_name == 'semiunet3plus_cbam':
        return SemiUNet3Plus_CBAM(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not implemented")