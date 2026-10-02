import torch
from torch import nn
from .decoder import Decoder
from models.twins_svt import alt_gvt_small
import torch.nn.functional as F
import logging


class Twins_BEAFormer(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.encoder = alt_gvt_small()
        dims = self.encoder.embed_dims[::-1]
        self.decoder = Decoder(dims)

        self.out_head1 = nn.Conv2d(dims[3], num_classes, 1)
        # self.out_head2 = nn.Conv2d(dims[2], num_classes, 1)
        # self.out_head3 = nn.Conv2d(dims[1], num_classes, 1)
        # self.out_head4 = nn.Conv2d(dims[0], num_classes, 1)

    def forward(self, x: torch.Tensor):
        H, W = x.shape[-2:]

        enc_feats, enc_gs = self.encoder(x)
        dec_feats, dec_gs = self.decoder(enc_feats)

        p1 = self.out_head1(dec_feats[0])
        p1 = F.interpolate(p1, size=(H, W), mode="bilinear")

        if self.training:
            # p2 = self.out_head2(dec_feats[1])
            # p3 = self.out_head3(dec_feats[2])
            # p4 = self.out_head4(dec_feats[3])

            # p2 = F.interpolate(p2, size=(H, W), mode="bilinear")
            # p3 = F.interpolate(p3, size=(H, W), mode="bilinear")
            # p4 = F.interpolate(p4, size=(H, W), mode="bilinear")

            gs = [
                F.interpolate(g, size=(H, W), mode="bilinear") for g in enc_gs
            ]

            return p1, gs

        return p1

