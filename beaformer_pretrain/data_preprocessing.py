from datasets import COD27K as cod27k_tools
from torch.utils.data import DataLoader

def cod27K():
    resize = (224, 224)
    src_root = "/mnt/folder_3/shy/data/data_zip/COD27K"
    dst_root = f"/mnt/folder_3/shy/data/COD27K_{resize[0]}x{resize[1]}"

    cod27k_tools.preprocessing(src_root, dst_root, resize)

    dataset = cod27k_tools.COD27K(dst_root)
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


if __name__ == "__main__":
    cod27K()