import torch
from torch import nn
import numpy as np
import numpy as np
import logging
import torch.nn.functional as F
from models.BEAFormer import BEAFormer


def load_model_state_dict(model: nn.Module, state_dict, strict = False):
    if isinstance(model, nn.DataParallel):
        model = model.module

    ret = model.load_state_dict(state_dict, strict)
    return ret


def get_model_state_dict(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    return model.state_dict()


def get_model(args):
    model = BEAFormer(num_classes=args.num_classes)

    if args.pretrained_fpath is not None:
        model_state_dict = torch.load(args.pretrained_fpath, map_location="cpu")
        ret = model.load_state_dict(model_state_dict, strict=False)
        msg = [f"Loaded model weights from {args.pretrained_fpath}"]
        if ret.missing_keys:
            msg.append(f"Missing keys: {ret.missing_keys}")
        if ret.unexpected_keys:
            msg.append(f"Unexpected keys: {ret.unexpected_keys}")

        logging.info("\n".join(msg))
    
    elif args.pretrained_backbone_fpath is not None:
        backbone_state_dict = torch.load(args.pretrained_backbone_fpath, map_location="cpu")
        new_state_dict = {}
        for k, v in backbone_state_dict.items():
            if ".mlp." in k:
                k1 = k.replace("mlp", "mlp1")
                k2 = k.replace("mlp", "mlp2")
                new_state_dict[k1] = v.clone()
                new_state_dict[k2] = v.clone()
            else:
                new_state_dict[k] = v
        
        ret = model.encoder.load_state_dict(new_state_dict, strict=False)
        msg = [f"Loaded model weights from {args.pretrained_backbone_fpath}"]
        if ret.missing_keys:
            msg.append(f"Missing keys: {ret.missing_keys}")
        if ret.unexpected_keys:
            msg.append(f"Unexpected keys: {ret.unexpected_keys}")

        logging.info("\n".join(msg))

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model


def get_bnd_label(label: torch.Tensor, kernel_size: int = 5):
    H, W = label.shape

    classes = torch.unique(label)
    bnd_label = torch.zeros((H, W), dtype=torch.long)
    pad = kernel_size // 2

    for c in classes:
        mask = (label == c).float()
        mask = mask.view(1, 1, H, W)

        erosion = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)

        boundary = mask - erosion
        boundary = boundary.view(H, W).long()

        bnd_label = torch.maximum(bnd_label, boundary)

    return bnd_label.long()