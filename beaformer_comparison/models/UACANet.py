import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.UACANet_lib.optim.losses import *
from models.UACANet_lib.modules.layers import *
from models.UACANet_lib.modules.context_module import *
from models.UACANet_lib.modules.attention_module import *
from models.UACANet_lib.modules.decoder_module import *

from models.UACANet_lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s
import logging

class UACANet(nn.Module):
    # res2net based encoder decoder
    def __init__(self,input_channel=3,num_classes=1, channels=256, output_stride=16, pretrained=True, pretrained_backbone_fpath=None):
        super(UACANet, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=False, output_stride=output_stride)
        if pretrained_backbone_fpath is not None:
            model_state_dict = torch.load(pretrained_backbone_fpath, map_location="cpu")
            ret = self.resnet.load_state_dict(model_state_dict, strict=False)

            msg = [f"Loaded model weights from {pretrained_backbone_fpath}"]
            if ret.missing_keys:
                msg.append(f"Missing keys: {ret.missing_keys}")
            if ret.unexpected_keys:
                msg.append(f"Unexpected keys: {ret.unexpected_keys}")

            logging.info("\n".join(msg))

        self.context2 = PAA_e(512, channels)
        self.context3 = PAA_e(1024, channels)
        self.context4 = PAA_e(2048, channels)

        self.decoder = PAA_d(channels, num_classes)

        self.attention2 = UACA(channels * 2, channels,num_classes)
        self.attention3 = UACA(channels * 2, channels,num_classes)
        self.attention4 = UACA(channels * 2, channels,num_classes)

        self.loss_fn = bce_iou_loss

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, sample):
        # x = sample['image']
        # if 'gt' in sample.keys():
        #     y = sample['gt']
        # else:
        #     y = None
        x=sample
        y=None
        if x.shape[1]==1:
            x=x.repeat(1,3,1,1)
            
        base_size = x.shape[-2:]
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)

        f5, a5 = self.decoder(x4, x3, x2)
        out5 = self.res(a5, base_size)

        f4, a4 = self.attention4(torch.cat([x4, self.ret(f5, x4)], dim=1), a5)
        out4 = self.res(a4, base_size)

        f3, a3 = self.attention3(torch.cat([x3, self.ret(f4, x3)], dim=1), a4)
        out3 = self.res(a3, base_size)

        _, a2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), a3)
        out2 = self.res(a2, base_size)


        if y is not None:
            loss5 = self.loss_fn(out5, y)
            loss4 = self.loss_fn(out4, y)
            loss3 = self.loss_fn(out3, y)
            loss2 = self.loss_fn(out2, y)

            loss = loss2 + loss3 + loss4 + loss5
            debug = [out5, out4, out3]
        else:
            loss = 0
            debug = []

        # return {'pred': out2, 'loss': loss, 'debug': debug}
        if self.training:
            return out2, out3, out4, out5
        
        return out2