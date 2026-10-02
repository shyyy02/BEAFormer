import torch
from torch import nn
from .decoder import Decoder
from models import PVTv2
import torch.nn.functional as F
import logging


class BEAFormer(nn.Module):
    def __init__(self, num_classes: int, encoder: str = "pvt_v2_b2"):
        super().__init__()

        self.encoder = getattr(PVTv2, encoder)()
        dims = self.encoder.embed_dims[::-1]
        self.decoder = Decoder(dims)

        self.out_head1 = nn.Conv2d(dims[3], num_classes, 1)
        # self.out_head2 = nn.Conv2d(dims[2], num_classes, 1)
        # self.out_head3 = nn.Conv2d(dims[1], num_classes, 1)
        # self.out_head4 = nn.Conv2d(dims[0], num_classes, 1)

    def freeze(self):
        unfrozen_params = []
        name_frozen_params = []
        for name, param in self.named_parameters():
            if not name.startswith("encoder") or "mlp2" in name or "routing" in name:
                param.requires_grad = True
                unfrozen_params.append(param)
            else:
                name_frozen_params.append(name)
                param.requires_grad = False

        logging.info(f"Frozen params: {name_frozen_params}")
        return unfrozen_params

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
                F.interpolate(g, size=(H, W), mode="bilinear") for g in enc_gs + dec_gs
            ]

            return p1, gs

        return p1

