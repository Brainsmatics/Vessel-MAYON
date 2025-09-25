import torch
import torch.nn as nn


class VYAF(nn.Module):
    def __init__(self, dimension=1, Channel1=1, Channel2=1):
        super(VYAF, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel_all = Channel1 + Channel2

        # 多尺度特征权重（保留加权特征）
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-5

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 自适应融合
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x1, x2 = x
        N1, C1, D1, H1, W1 = x1.size()
        N2, C2, D2, H2, W2 = x2.size()

        # 通道权重融合（保留加权特征）
        w = torch.sigmoid(self.w[:(C1 + C2)])
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        # 加权特征（保持不变）
        x1 = (weight[:C1] * x1.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x2 = (weight[C1:] * x2.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        # 拼接特征
        fused = torch.cat([x1, x2], dim=self.d)

        # 仅使用空间注意力
        avg_out = torch.mean(fused, dim=1, keepdim=True)
        max_out, _ = torch.max(fused, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        refined = fused * spatial_att

        # 自适应残差连接
        output = self.alpha * refined + self.beta * fused

        return output



class ResConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.05, inplace=True)  # 修改为LeakyReLU
        )
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class MDCunit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        branch_channels = out_channels // 4
        out_channels = branch_channels * 4

        self.axial = nn.Conv3d(in_channels, branch_channels,
                               (kernel_size[0], kernel_size[1], 1),
                               padding=(kernel_size[0] // 2, kernel_size[1] // 2, 0))
        self.sagittal = nn.Conv3d(in_channels, branch_channels,
                                  (kernel_size[0], 1, kernel_size[2]),
                                  padding=(kernel_size[0] // 2, 0, kernel_size[2] // 2))
        self.coronal = nn.Conv3d(in_channels, branch_channels,
                                 (1, kernel_size[1], kernel_size[2]),
                                 padding=(0, kernel_size[1] // 2, kernel_size[2] // 2))
        self.cubic = nn.Conv3d(in_channels, branch_channels, kernel_size,
                               padding=tuple(k // 2 for k in kernel_size))
        self.bn = nn.InstanceNorm3d(4 * branch_channels)

        self.act = nn.LeakyReLU(0.05, inplace=True)  # 修改为LeakyReLU
        self.fusion = nn.Conv3d(4 * branch_channels, out_channels, 1)
        for conv in [self.axial, self.sagittal, self.coronal, self.cubic, self.fusion]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='leaky_relu')
            nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        axial = self.axial(x)
        sagittal = self.sagittal(x)
        coronal = self.coronal(x)
        cubic = self.cubic(x)

        x = torch.cat([axial, sagittal, coronal, cubic], dim=1)
        x = self.bn(x)
        x = self.act(x)
        return self.fusion(x)


class MDCblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = MDCunit(in_channels, out_channels, kernel_size)
        self.conv2 = MDCunit(out_channels, out_channels, kernel_size)

        self.residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class MAYON(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DirectionalOnlyEncoder()
        self.decoder = DirectionalOnlyDecoderWithFFM()

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


class DirectionalOnlyDecoderWithFFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.ModuleList([
            nn.ConvTranspose3d(320, 320, (1, 2, 2), stride=(1, 2, 2)),
            nn.ConvTranspose3d(320, 256, (1, 2, 2), stride=(1, 2, 2)),
            nn.ConvTranspose3d(256, 128, (2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(128, 64, (2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        ])

        self.ffm_fusions = nn.ModuleList([
            VYAF(dimension=1, Channel1=320, Channel2=320),
            VYAF(dimension=1, Channel1=256, Channel2=256),
            VYAF(dimension=1, Channel1=128, Channel2=128),
            VYAF(dimension=1, Channel1=64, Channel2=64),
            VYAF(dimension=1, Channel1=32, Channel2=32)
        ])

        self.post_fusion_blocks = nn.ModuleList([
            ResConv3d(640, 320),
            ResConv3d(512, 256),
            MDCblock(256, 128),
            MDCblock(128, 64),
            MDCblock(64, 32)
        ])

        self.seg_output = nn.Conv3d(32, 2, kernel_size=1)
        nn.init.normal_(self.seg_output.weight, std=0.01)


    def forward(self, encoder_outputs):
        x = encoder_outputs[-1]

        for i in range(5):
            x = self.upsample[i](x)
            x = self.ffm_fusions[i]([x, encoder_outputs[-i - 2]])
            x = self.post_fusion_blocks[i](x)

        return self.seg_output(x)


class DirectionalOnlyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = ResConv3d(1, 32)
        self.stage1 = ResConv3d(32, 64)
        self.stage2 = MDCunit(64, 128)
        self.stage3 = MDCunit(128, 256)
        self.stage4 = ResConv3d(256, 320)
        self.stage5 = ResConv3d(320, 320)

        self.downsample = nn.ModuleList([
            nn.MaxPool3d((2, 2, 2)),
            nn.MaxPool3d((2, 2, 2)),
            nn.MaxPool3d((2, 2, 2)),
            nn.MaxPool3d((1, 2, 2)),
            nn.MaxPool3d((1, 2, 2))
        ])

    def forward(self, x):
        features = []
        x = self.stage0(x)
        features.append(x)

        for i in range(1, 6):
            x = self.downsample[i - 1](x)
            x = getattr(self, f'stage{i}')(x)
            features.append(x)

        return features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAYON().to(device)
    print("参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    input_tensor = torch.randn(2, 1, 56, 192, 160).to(device)
    output = model(input_tensor)
    print("输入尺寸:", input_tensor.shape)
    print("输出尺寸:", output.shape)