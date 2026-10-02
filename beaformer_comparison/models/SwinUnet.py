from models.SwinUnet_lib.vision_transformer import SwinUnet
from models.SwinUnet_lib.config import get_config
import argparse

def swinunet(num_classes, pretrained_backbone_fpath, input_channel=3):
    model = SwinUnet(img_size=224,input_channel=input_channel, num_classes=num_classes)
    model.load_from(pretrained_backbone_fpath)
    return model