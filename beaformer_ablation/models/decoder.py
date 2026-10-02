import torch
from torch import nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = torch.cat([x, z], dim=1)
        x = self.norm(x)
        x = self.conv(x)

        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hid_dim: int):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hid_dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(hid_dim)
        self.dwconv = nn.Conv2d(hid_dim, hid_dim, 3, 1, 1, groups=hid_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hid_dim, dim, 1)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.norm(x)
        x = x + self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        
        return x


# Parallel Depthwise Convolution
class PDWConv(nn.Module):
    def __init__(self, dim: int, kernel_list: tuple | list):
        super().__init__()
        self.dwconvs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                )
                for k in kernel_list
            ]
        )

    def forward(self, x: torch.Tensor):
        return sum([dwconv(x) for dwconv in self.dwconvs])


class ConvBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_list: tuple | list,
        mlp_ratio: float,
    ):
        super().__init__()
        self.token_mixer = PDWConv(dim, kernel_list)
        self.routing = nn.Sequential(
            nn.Conv2d(dim, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        hid_dim = int(dim * mlp_ratio)
        self.mlp1 = MLP(dim, hid_dim)
        self.mlp2 = MLP(dim, hid_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.token_mixer(x)
        
        g = self.routing(x)
        x1 = self.mlp1(x)
        x2 = self.mlp2(x)
        x = x + g * x1 + (1 - g) * x2

        return x, g


class Decoder(nn.Module):
    def __init__(
        self,
        dims: tuple | list,
        depths: tuple | list = (3, 3, 3),
        kernel_list: tuple | list = (3, 7),
        mlp_ratio: float = 2.5,
    ):
        super().__init__()

        self.fusions = nn.ModuleList(
            [Fusion(dims[i] + dims[i + 1], dims[i + 1]) for i in range(3)]
        )

        self.stages = nn.ModuleList()
        for i in range(3):
            blocks = nn.ModuleList(
                [
                    ConvBlock(dims[i + 1], kernel_list, mlp_ratio)
                    for _ in range(depths[i])
                ]
            )
            self.stages.append(blocks)

    def forward(self, enc_feats: list[torch.Tensor]):
        [e1, e2, e3, e4] = enc_feats
        gs = []

        d4 = e4

        d3 = self.fusions[0](d4, e3)
        for block in self.stages[0]:
            d3, g = block(d3)
            gs.append(g)

        d2 = self.fusions[1](d3, e2)
        for block in self.stages[1]:
            d2, g = block(d2)
            gs.append(g)

        d1 = self.fusions[2](d2, e1)
        for block in self.stages[2]:
            d1, g = block(d1)
            gs.append(g)

        return [d1, d2, d3, d4], gs
