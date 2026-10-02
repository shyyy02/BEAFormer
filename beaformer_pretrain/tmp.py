import torch 
from torchinfo import summary
import os
import nibabel as nib
import numpy as np
from models.BEAFormer import BEAFormer
from collections import Counter
import cv2
from tqdm import tqdm
from PIL import Image

# device = torch.device('cuda:0')
# model = BEAFormer(num_classes=1)
# state_dict = torch.load("/home/shy/lab3_v4_2/outputs/hqseg44k/run1/final_model_state.pth", map_location="cpu")
# model.load_state_dict(state_dict, strict=False)

# state_dict = model.encoder.state_dict()
# torch.save(state_dict, "/home/shy/lab3_v4_2/outputs/hqseg44k/run1/backbone.pth")



# summary(model, input_size=(1, 3, 224, 224), device=device) 


# for k, v in model.freeze():
#     print(k)


folder = "/mnt/folder_3/shy/lab3_v5_2_3/outputs/hqseg44k/run2"
path = os.path.join(folder, "model_state.pth")
save_path = os.path.join(folder, "backbone.pth")

model_state = torch.load(path, map_location="cpu")
model_state = {
    k.split(".", 1)[1]: v
    for k, v in model_state.items()
    if not k.startswith("decoder")
}
torch.save(model_state, save_path)