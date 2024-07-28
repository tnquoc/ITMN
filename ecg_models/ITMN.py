import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super(MambaBlock, self).__init__()
        self.mixer = Mamba(d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.mixer(self.norm(x))

        return x


class BaseInceptionBlock(nn.Module):
    def __init__(self, d_model):
        super(BaseInceptionBlock, self).__init__()
        dim = d_model // 4
        self.bottleneck = nn.Conv1d(d_model, dim, kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv1d(dim, dim, kernel_size=39, stride=1, padding=19, bias=False)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=19, stride=1, padding=9, bias=False)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=9, stride=1, padding=4, bias=False)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.conv1 = nn.Conv1d(d_model, dim, kernel_size=1, stride=1, bias=False)

        self.bn = nn.BatchNorm1d(d_model, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.bottleneck(x)
        output4 = self.conv4(output)
        output3 = self.conv3(output)
        output2 = self.conv2(output)

        output1 = self.maxpool(x)
        output1 = self.conv1(output1)

        x_out = self.relu(self.bn(torch.cat((output1, output2, output3, output4), dim=1)))
        return x_out


class ISSMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ISSMBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
        self.inception_block = BaseInceptionBlock(out_channels)
        self.mamba_block = MambaBlock(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x1 = self.inception_block(x)
        x2 = self.relu(self.mamba_block(x.transpose(-1, -2)).transpose(-1, -2))
        x = x1 + x2

        return x


class ITMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ITMBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
        self.inception_block = BaseInceptionBlock(out_channels)
        self.mamba_block = MambaBlock(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x1 = self.inception_block(x)
        x2 = self.relu(self.mamba_block(x.transpose(-1, -2)).transpose(-1, -2))
        x = x1 + x2

        return x


class ITMN(nn.Module):
    def __init__(self, d_model, n_classes=2):
        """Full Mamba model."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(12, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
        )
        self.layers = nn.Sequential(
            ITMBlock(d_model, d_model),
            ITMBlock(d_model, d_model),
            nn.MaxPool1d(2, 2),
            ITMBlock(d_model, d_model),
            ITMBlock(d_model, d_model),
            nn.MaxPool1d(2, 2),
            ITMBlock(d_model, 2 * d_model),
        )

        self.classifier = nn.Linear(2 * d_model, n_classes)

        self.apply(init_weights)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        x = self.layers(x)
        x = x.mean(dim=-1)

        x = self.classifier(x)

        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


if __name__ == '__main__':
    device = 'cuda'
    x = torch.rand((4, 1000, 12)).to(device)
    model = ITMN(d_model=64, n_classes=5).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(x)
    print(out.shape)
