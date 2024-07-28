import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


class ECGMamba(nn.Module):
    def __init__(self, d_model, n_layers=4, n_classes=5):
        """Full Mamba model."""
        super().__init__()
        self.encoder = nn.Conv1d(12, d_model, kernel_size=1)
        self.layers = nn.ModuleList([ResidualBlock(d_model) for i in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, n_classes)
        )

        self.apply(init_weights)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        x = x.transpose(-1, -2)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        x = x.transpose(-1, -2)

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


class ResidualBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mixer = Mamba(d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        #  x : (B, L, D)

        #  output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    # def step(self, x, cache):
    #     #  x : (B, D)
    #     #  cache : (h, inputs)
    #     # h : (B, ED, N)
    #     #  inputs: (B, ED, d_conv-1)
    #
    #     #  output : (B, D)
    #     #  cache : (h, inputs)
    #
    #     output, cache = self.mixer.step(self.norm(x), cache)
    #     output = output + x
    #     return output, cache


#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


if __name__ == '__main__':
    device = 'cuda'
    x = torch.rand((4, 1000, 12)).to(device)
    model = ECGMamba(d_model=128, n_classes=5).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(x)
    print(out.shape)
