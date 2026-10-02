from torch import nn
import torch
import logging
from utils import load_model_state_dict
import models.MISSFormer as MISSFormer_lib
import models.PVT_EMCAD as PVT_EMCAD_lib
import models.PraNet as PraNet_lib
import models.U_KAN as U_KAN_lib
import models.UTANet as UTANet_lib
import models.MobileUViT as MobileUViT_lib
import models.RollingUnet as RollingUnet_lib
# import models.Swin_umamba as Swin_umamba_lib
import models.Perspective_Unet as Perspective_Unet_lib
import models.SwinUnet as SwinUnet_lib
import models.UACANet as UACANet_lib
import models.U_Net as U_Net_lib
import models.CA_Net as CA_Net_lib
import models.MedT as MedT_lib
import models.FCBFormer as FCBFormer_lib
import models.H2Former as H2Former_lib
import models.CFFormer as CFFormer_lib
import models.LGMSNet as LGMSNet_lib
import models.UCTransNet as UCTransNet_lib
import models.TransUNet as TransUNet_lib
import models.PMRNet as PMRNet_lib

# 以下两个在运行时必须注释掉一个
# import models.U_RWKV as U_RWKV_lib
# import models.Zig_RiR as Zig_RiR_lib


def get_model(args):
    model = globals()[args.model](**vars(args))

    if args.pretrained_fpath is not None:
        model_state_dict = torch.load(args.pretrained_fpath, map_location="cpu")
        ret = load_model_state_dict(model, model_state_dict)
        msg = [f"Loaded model weights from {args.pretrained_fpath}"]
        if ret.missing_keys:
            msg.append(f"Missing keys: {ret.missing_keys}")
        if ret.unexpected_keys:
            msg.append(f"Unexpected keys: {ret.unexpected_keys}")

        logging.info("\n".join(msg))

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model


# class Zig_RiR(nn.Module):
#     def __init__(self, num_classes: int, **kwargs):
#         super().__init__()
#         self.net = Zig_RiR_lib.zig_rir(num_classes=num_classes)

#     def forward(self, x: torch.Tensor):
#         return self.net(x)


class PMRNet(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.net = PMRNet_lib.PMRNet(num_classes=num_classes)
    
    def forward(self, x: torch.Tensor):
        return self.net(x)
    

class TransUNet(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone_fpath: str = None, **kwargs):
        super().__init__()
        self.net = TransUNet_lib.TransUnet(num_classes=num_classes, pretrained_backbone_fpath=pretrained_backbone_fpath)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    

class UCTransNet(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = UCTransNet_lib.UCTransNet(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    

# class U_RWKV(nn.Module):
#     def __init__(self, num_classes: int, **kwargs):
#         super().__init__()
#         self.net = U_RWKV_lib.u_rwkv(num_classes=num_classes)

#     def forward(self, x: torch.Tensor):
#         return self.net(x)
    

class LGMSNet(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = LGMSNet_lib.lgmsnet(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    

class CFFormer(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone_fpath: str, **kwargs):
        super().__init__()
        self.net = CFFormer_lib.cfformer(num_classes=num_classes, pretrained_backbone_fpath=pretrained_backbone_fpath)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    

class H2Former(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = H2Former_lib.h2former(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    

class FCBFormer(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone_fpath: str, **kwargs):
        super().__init__()
        self.net = FCBFormer_lib.fcbformer(
            num_classes=num_classes, pretrained_backbone_fpath=pretrained_backbone_fpath
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
    

class MedT(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = MedT_lib.medt(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class CA_Net(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = CA_Net_lib.ca_net(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class U_Net(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = U_Net_lib.U_Net(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class UACANet_L(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone_fpath: str, **kwargs):
        super().__init__()
        self.net = UACANet_lib.UACANet(
            num_classes=num_classes, pretrained_backbone_fpath=pretrained_backbone_fpath
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class SwinUnet(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone_fpath: str, **kwargs):
        super().__init__()
        self.net = SwinUnet_lib.swinunet(
            num_classes=num_classes, pretrained_backbone_fpath=pretrained_backbone_fpath
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Perspective_Unet(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = Perspective_Unet_lib.perspective_unet(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)


# class Swin_umamba(nn.Module):
#     def __init__(self, num_classes: int, pretrained_backbone_fpath: str, **kwargs):
#         super().__init__()
#         self.net = Swin_umamba_lib.swin_umamba(
#             num_classes=num_classes, pretrained_backbone_fpath=pretrained_backbone_fpath
#         )

#     def forward(self, x: torch.Tensor):
#         return self.net(x)


class RollingUnet_L(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = RollingUnet_lib.Rolling_Unet_L(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MobileUViT_L(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = MobileUViT_lib.mobileuvit_l(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class UTANet(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone_fpath: str, **kwargs):
        super().__init__()
        self.net = UTANet_lib.UTANet(
            n_classes=num_classes,
            pretrained_backbone_fpath=pretrained_backbone_fpath,
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class U_KAN(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.net = U_KAN_lib.UKAN(num_classes=num_classes, img_size=224)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MISSFormer(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone_fpath: str, **kwargs):
        super().__init__()
        self.net = MISSFormer_lib.MISSFormer(
            num_classes=num_classes,
            pretrained_backbone_fpath=pretrained_backbone_fpath,
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class PVT_EMCAD(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone_fpath: str, **kwargs):
        super().__init__()

        self.net = PVT_EMCAD_lib.EMCADNet(
            num_classes=num_classes, pretrained_backbone_fpath=pretrained_backbone_fpath
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class PraNet(nn.Module):
    def __init__(self, pretrained_backbone_fpath: str, **kwargs):
        super().__init__()

        self.net = PraNet_lib.PraNet(
            pretrained_backbone_fpath=pretrained_backbone_fpath,
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
