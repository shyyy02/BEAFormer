from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import logging
from datasets.BUSI import BUSI
from datasets.BUSBRA import BUSBRA
from datasets.BUS_UC import BUS_UC
from datasets.ISIC2017 import ISIC2017
from datasets.ISIC2018 import ISIC2018
from datasets.PolypDB_WLI import PolypDB_WLI
from datasets.CHAOS import CHAOS
from datasets.BRISC2025 import BRISC2025
from datasets.BKAI import BKAI
from datasets.KvasirSEG import KvasirSEG
from datasets.QaTa_COV19_v2 import QaTa_COV19_v2
from datasets.QaTa_COV19_v1 import QaTa_COV19_v1
from datasets.BTMRI import BTMRI
from datasets.NuSeC import NuSeC
from datasets.GlaS import GlaS
from metrics import metrics_binary
import time
from utils import get_model
import os
import sys
from sklearn.model_selection import KFold

def infer(
    args,
    model: nn.Module,
    test_db: Dataset,
    metrics_fn,
):

    torch.cuda.reset_peak_memory_stats()
    total_time = 0
    num_slices = 0
    test_loader = DataLoader(test_db, batch_size=1, shuffle=False, pin_memory=True)
    model.eval()

    num_samples = len(test_db)
    metrics = defaultdict(float)

    with torch.no_grad():
        logging.info(f"Loaded {len(test_loader)} samples for inference")
        for i, batch in enumerate(test_loader):
            logging.info(f"Inferring sample {i}...")

            images = batch["image"]
            labels = batch["label"]

            dims = len(images.shape)
            assert dims in [4, 5], f"Invalid batch shape: {images.shape}"

            if dims == 5:
                images = images.squeeze(0)

            N = images.shape[0]

            # warmup
            for i in range(0, min(10, N)):
                image = images[i : i + 1].cuda(non_blocking=True)
                pred = model(image)

            preds = []
            for i in range(0, N):
                image = images[i : i + 1].cuda(non_blocking=True)

                torch.cuda.synchronize()
                start_time = time.perf_counter()

                pred = model(image)

                torch.cuda.synchronize()
                end_time = time.perf_counter()

                total_time += end_time - start_time
                pred = pred.cpu().detach()
                preds.append(pred)
            preds = torch.cat(preds, dim=0)

            preds = (
                preds.argmax(dim=1)
                if preds.shape[1] > 1
                else (preds > 0).long().squeeze(1)
            )

            if dims == 5:
                preds = preds.unsqueeze(0)

            case_metrics = metrics_fn(preds.squeeze(0), labels.squeeze(0))
            for k, v in case_metrics.items():
                metrics[k] += v
                logging.info(f"{k}: {v:.6f}")
            logging.info("")

            num_slices += N

    for k in metrics:
        metrics[k] /= num_samples

    logging.info("Inference completed")
    for k, v in metrics.items():
        logging.info(f"{k}: {v:.6f}")

    logging.info("")
    logging.info(f"Throughput: {num_slices / total_time:.2f} samples/s")
    logging.info(f"MMA: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    logging.info(f"MMR: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")

    return metrics


def infer_polypdb_wli(args):

    model = get_model(args)

    test_db = PolypDB_WLI(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_bkai(args):

    model = get_model(args)

    test_db = BKAI(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_kvasirseg(args):

    model = get_model(args)

    test_db = KvasirSEG(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_busi(args):

    model = get_model(args)

    test_db = BUSI(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_busbra(args):

    model = get_model(args)

    test_db = BUSBRA(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_bus_uc(args):

    model = get_model(args)

    test_db = BUS_UC(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_brisc2025(args):

    model = get_model(args)

    test_db = BRISC2025(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)
    

def infer_isic2017(args):

    model = get_model(args)

    test_db = ISIC2017(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_isic2018(args):

    model = get_model(args)

    test_db = ISIC2018(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_chaos(args):
    case_ids = sorted(os.listdir(os.path.join(args.root, "slices", "images")))
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=1234)
    outputs_dir = args.outputs_dir

    metrics_tot = defaultdict(float)
    for i, (train_ids, valid_ids) in enumerate(kf.split(case_ids)):
        valid_case_ids = [case_ids[j] for j in valid_ids]

        args.outputs_dir = os.path.join(outputs_dir, f"fold_{i}")
        args.pretrained_fpath = os.path.join(args.outputs_dir, "best_model.pth")

        log_fpath = os.path.join(args.outputs_dir, f"val_log.txt")
        logging.basicConfig(
            filename=log_fpath,
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        model = get_model(args)

        valid_db = CHAOS(root=args.root, case_ids=valid_case_ids, split="valid")
        metrics_fn = metrics_binary
        metrics = infer(args, model, valid_db, metrics_fn)
        
        for k, v in metrics.items():
            metrics_tot[k] += v

    log_fpath = os.path.join(args.outputs_dir, f"{args.mode}_log.txt")
    logging.basicConfig(
        filename=log_fpath,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info("Inference completed")
    
    for k, v in metrics_tot.items():
        logging.info(f"{k}: {(v / args.k_folds):.6f}")
    

def infer_qata_cov19_v2(args):

    model = get_model(args)

    test_db = QaTa_COV19_v2(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_qata_cov19_v1(args):

    model = get_model(args)

    test_db = QaTa_COV19_v1(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_btmri(args):

    model = get_model(args)

    test_db = BTMRI(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_nusec(args):

    model = get_model(args)

    test_db = NuSeC(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)


def infer_glas(args):

    model = get_model(args)

    test_db = GlaS(root=args.root, split="test")
    metrics_fn = metrics_binary

    infer(args, model, test_db, metrics_fn)