from datasets import BUSI as busi_tools
from datasets import BUSBRA as busbra_tools
from datasets import ISIC2017 as isic2017_tools
from datasets import PolypDB_WLI as polypdb_wli_tools
from datasets import CHAOS as chaos_tools
from datasets import ISIC2018 as isic2018_tools
from datasets import BRISC2025 as brisc2025_tools
from datasets import BKAI as bkai_tools
from datasets import KvasirSEG as kvasirseg_tools
from datasets import QaTa_COV19_v2 as qata_cov19_v2_tools
from datasets import QaTa_COV19_v1 as qata_cov19_v1_tools
from datasets import BTMRI as btmri_tools
from datasets import NuSeC as nusec_tools
from datasets import GlaS as glas_tools
from torch.utils.data import DataLoader


def polypdb_wli():
    resize = (224, 224)
    split_ratio = (0.7, 0.15, 0.15)
    src_root = "/mnt/folder_3/shy/data/data_zip/PolypDB-WLI"
    dst_root = f"/mnt/folder_3/shy/data/PolypDB-WLI_{resize[0]}x{resize[1]}"

    polypdb_wli_tools.preprocessing(src_root, dst_root, split_ratio, resize)

    dataset = polypdb_wli_tools.PolypDB_WLI(dst_root, "train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def bkai():
    resize = (224, 224)
    split_ratio = (0.7, 0.15, 0.15)
    src_root = "/mnt/folder_3/shy/data/data_zip/BKAI"
    dst_root = f"/mnt/folder_3/shy/data/BKAI_{resize[0]}x{resize[1]}"

    bkai_tools.preprocessing(src_root, dst_root, split_ratio, resize)

    dataset = bkai_tools.BKAI(dst_root, "train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def kvasirseg():
    resize = (224, 224)
    src_root = "/mnt/folder_3/shy/data/data_zip/Kvasir-SEG"
    dst_root = f"/mnt/folder_3/shy/data/KvasirSEG_{resize[0]}x{resize[1]}"

    kvasirseg_tools.preprocessing(src_root, dst_root, resize)

    dataset = kvasirseg_tools.KvasirSEG(dst_root, "train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def busi():
    resize = (224, 224)
    split_ratio = (0.7, 0.15, 0.15)
    src_root = "/mnt/folder_3/shy/data/data_zip/BUSI"
    dst_root = f"/mnt/folder_3/shy/data/BUSI_{resize[0]}x{resize[1]}"

    busi_tools.preprocessing(src_root, dst_root, split_ratio, resize)

    dataset = busi_tools.BUSI(dst_root, "train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def busbra():
    resize = (224, 224)
    split_ratio = (0.7, 0.15, 0.15)
    src_root = "/mnt/folder_3/shy/data/data_zip/BUSBRA"
    dst_root = f"/mnt/folder_3/shy/data/BUSBRA_{resize[0]}x{resize[1]}"

    busbra_tools.preprocessing(src_root, dst_root, split_ratio, resize)

    dataset = busbra_tools.BUSBRA(dst_root, "train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def brisc2025():
    resize = (224, 224)
    src_root = "/mnt/folder_3/shy/data/data_zip/BRISC2025"
    dst_root = f"/mnt/folder_3/shy/data/BRISC2025_{resize[0]}x{resize[1]}"

    brisc2025_tools.preprocessing(src_root, dst_root, resize)

    dataset = brisc2025_tools.BRISC2025(dst_root, "train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def isic2017():
    resize = (224, 224)
    src_root = "/mnt/folder_3/shy/data/data_zip/ISIC2017_Task1"
    dst_root = f"/mnt/folder_3/shy/data/ISIC2017_{resize[0]}x{resize[1]}"

    isic2017_tools.preprocessing(src_root, dst_root, resize)

    dataset = isic2017_tools.ISIC2017(dst_root, split="train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def isic2018():
    resize = (224, 224)
    src_root = "/mnt/folder_3/shy/data/data_zip/ISIC2018_Task1"
    dst_root = f"/mnt/folder_3/shy/data/ISIC2018_{resize[0]}x{resize[1]}"

    isic2018_tools.preprocessing(src_root, dst_root, resize)

    dataset = isic2018_tools.ISIC2018(dst_root, split="train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def chaos():
    resize = (224, 224)
    src_root = "/mnt/folder_3/shy/data/data_zip/CHAOS"
    dst_root = f"/mnt/folder_3/shy/data/CHAOS_{resize[0]}x{resize[1]}"

    chaos_tools.preprocessing(src_root, dst_root, resize)

    train_db = chaos_tools.CHAOS(dst_root, case_ids=[1, 2], split="train")
    train_loader = DataLoader(train_db, batch_size=2)
    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        print(images.shape, images.min(), images.max(), images.dtype)
        print(labels.shape, labels.min(), labels.max(), labels.dtype)
        break

    valid_db = chaos_tools.CHAOS(dst_root, case_ids=[1, 2], split="valid")
    valid_loader = DataLoader(valid_db, batch_size=1)
    for batch in valid_loader:
        images = batch["image"]
        labels = batch["label"]
        print(images.shape, images.min(), images.max(), images.dtype)
        print(labels.shape, labels.min(), labels.max(), labels.dtype)
        break


def qata_cov19_v2():
    resize = (224, 224)
    src_root = "/mnt/folder_3/shy/data/data_zip/QaTa-COV19-v2"
    dst_root = f"/mnt/folder_3/shy/data/QaTa_COV19_v2_{resize[0]}x{resize[1]}"

    qata_cov19_v2_tools.preprocessing(src_root, dst_root, resize)

    dataset = qata_cov19_v2_tools.QaTa_COV19_v2(dst_root, split="train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def qata_cov19_v1():
    resize = (224, 224)
    split_ratio = (0.7, 0.15, 0.15)
    src_root = "/mnt/folder_3/shy/data/data_zip/QaTa-COV19-v1"
    dst_root = f"/mnt/folder_3/shy/data/QaTa_COV19_v1_{resize[0]}x{resize[1]}"

    qata_cov19_v1_tools.preprocessing(src_root, dst_root, split_ratio, resize)

    dataset = qata_cov19_v1_tools.QaTa_COV19_v1(dst_root, split="train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def btmri():
    resize = (224, 224)
    split_ratio = (0.7, 0.15, 0.15)
    clip_upper = 1976.0 # 99% 分位数
    src_root = "/mnt/folder_3/shy/data/data_zip/BTMRI/mat"
    dst_root = f"/mnt/folder_3/shy/data/BTMRI_{resize[0]}x{resize[1]}"
    
    btmri_tools.preprocessing(src_root, dst_root, split_ratio, resize, clip_upper)

    dataset = btmri_tools.BTMRI(dst_root, split="train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def nusec():
    resize = (224, 224)
    src_root = "/mnt/folder_3/shy/data/data_zip/NuSeC"
    dst_root = f"/mnt/folder_3/shy/data/NuSeC_{resize[0]}x{resize[1]}"
    
    nusec_tools.preprocessing(src_root, dst_root, resize)

    dataset = nusec_tools.NuSeC(dst_root, split="train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


def glas():
    resize = (224, 224)
    src_root = "/mnt/folder_3/shy/data/data_zip/GlaS"
    dst_root = f"/mnt/folder_3/shy/data/GlaS_{resize[0]}x{resize[1]}"
    
    glas_tools.preprocessing(src_root, dst_root, resize)

    dataset = glas_tools.GlaS(dst_root, split="train")
    data_loader = DataLoader(dataset, batch_size=2)
    for sample in data_loader:
        image = sample["image"]
        label = sample["label"]
        print(image.dtype, image.shape, image.min(), image.max())
        print(label.dtype, label.shape, label.min(), label.max())
        break


if __name__ == "__main__":
    glas()
