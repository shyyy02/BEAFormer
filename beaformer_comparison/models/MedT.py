from models.MedT_lib.axialnet import MedT

def medt(num_classes, input_channel=3):
    return MedT(img_size=224, input_channel=input_channel, num_classes=num_classes)