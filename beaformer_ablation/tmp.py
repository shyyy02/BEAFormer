import torch 
from torchinfo import summary
import os
import nibabel as nib
import numpy as np
from models.BEAFormer import BEAFormer
import utils

device = torch.device('cuda:0')
model = BEAFormer(1)
# model_dict = torch.load("/home/shy/backbone/beaformer_b2_deepsup.pth")
# ret = model.load_state_dict(model_dict)
# print(ret)

# print(type(model.backbone))
# backbone_state_dict = torch.load("/home/shy/backbone/pvt_v2_b2.pth", map_location="cpu")
# new_state_dict = {}
# for k, v in backbone_state_dict.items():
#     if ".mlp." in k:
#         k1 = k.replace("mlp", "mlp1")
#         k2 = k.replace("mlp", "mlp2")
#         new_state_dict[k1] = v.clone()
#         new_state_dict[k2] = v.clone()
#     else:
#         new_state_dict[k] = v

# ret = model.encoder.load_state_dict(new_state_dict, strict=False)

# print(ret)


summary(model, input_size=(1, 3, 224, 224), device=device) 

# model_dict = torch.load("/home/shy/lab3_v5/outputs/polyp/run1/checkpoint.pth")
# model_dict = model_dict["model_state_dict"]
# torch.save(model_dict, "/home/shy/lab3_v5/outputs/polyp/run1/model.pth")
# print(model_dict.keys())

# for mod in model.named_modules():
#     print(mod)
# print(len([p for p in model.net.backbone_parameters()]))
# print(len(model.net.non_backbone_parameters()))

