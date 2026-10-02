import torch 
from torchinfo import summary
from models.model_lib import *
import os
import nibabel as nib
import numpy as np

# print(torch.cuda.device_count())
device = torch.device('cuda:0')
model = PMRNet(num_classes=1)
summary(model, input_size=(1, 3, 224, 224), device=device) 

# for mod in model.named_modules():
#     print(mod)
# print(len([p for p in model.net.backbone_parameters()]))
# print(len(model.net.non_backbone_parameters()))


# src_img_fpath = "/home/shy/data/data_zip/Task09_Spleen/labelsTr/spleen_10.nii.gz"
# img = nib.load(src_img_fpath)

# arr = img.get_fdata(dtype=np.float32)

# print("\n=== NumPy Array Info ===")
# print("Shape:", arr.shape)
# print("Dtype:", arr.dtype)
# print("Min:", np.nanmin(arr))
# print("Max:", np.nanmax(arr))
# print("Mean:", np.nanmean(arr))
# print("Std:", np.nanstd(arr))

# # NaN / Inf 检查
# print("Contains NaN:", np.isnan(arr).any())
# print("Contains +Inf/-Inf:", np.isinf(arr).any())


# # 查看 3D/4D 结构
# print("Dimensions:", arr.ndim)

# print("========================\n")


# # --- 查看 Header 元数据 ---
# hdr = img.header
# print("Header info:")
# print(hdr)

# # --- 查看 slope / intercept ---
# slope = hdr.get('scl_slope', 1.0)
# inter = hdr.get('scl_inter', 0.0)
# print("\nscl_slope:", slope)
# print("scl_inter:", inter)

# # --- 读取 raw 数据（不做转换） ---
# raw = img.dataobj[:]     # 原始存储数据
# print("\nRaw dtype:", raw.dtype)

# # --- 自动转换后的数据（get_fdata） ---
# auto_hu = img.get_fdata(dtype=np.float32)
# print("Converted dtype:", auto_hu.dtype)

# # --- 手动转换 raw → HU ---
# manual_hu = raw * (slope if slope != 0 else 1.0) + inter

# # --- 验证是否一致 ---
# diff = np.abs(auto_hu - manual_hu).max()
# print("\nMax difference between auto and manual HU:", diff)

# if diff == 0:
#     print("✔ get_fdata() == raw * slope + intercept （完全一致）")
# else:
#     print("⚠ 有差异，需要进一步检查")