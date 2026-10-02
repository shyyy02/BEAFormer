from models.TransUNet_lib.vit_seg_modeling import VisionTransformer as ViT_seg
from models.TransUNet_lib.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torch import nn
import numpy as np

class TransUnet(nn.Module):
    def __init__(self, input_channel=3, pretrained_backbone_fpath=None, num_classes=1):
        super(TransUnet, self).__init__()
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.pretrained_path = pretrained_backbone_fpath
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        self.net = ViT_seg(config_vit, img_size=224, num_classes=num_classes).cuda()

        if pretrained_backbone_fpath is not None:
            weights = np.load(pretrained_backbone_fpath)
            self.net.load_from(weights)

    def forward(self, x):
        return self.net(x)


def transunet(input_channel=3, num_classes=1):
    return TransUnet(input_channel=input_channel, num_classes=num_classes)