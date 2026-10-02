import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import os
import sys
from tensorboardX import SummaryWriter
import logging
from utils import *
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
import random
from scheduler import WarmupCosineAnnealingLR
from loss import *
from metrics import metrics_binary
from collections import defaultdict
from sklearn.model_selection import KFold


def eval(
    args,
    model: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    writer: SummaryWriter,
    metrics_fn,
    primary_metric: str,
):
    model.eval()

    num_samples = 0
    metrics = defaultdict(float)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"]
            labels = batch["label"]

            dims = len(images.shape)
            assert dims in [4, 5], f"Invalid batch shape: {images.shape}"

            if dims == 5:
                B, D, C, H, W = images.shape
                images = images.reshape(-1, C, H, W)
            else:
                B, C, H, W = images.shape

            N = images.shape[0]

            preds = []
            for i in range(0, N, args.batch_size):
                image = images[i : i + args.batch_size].cuda()
                pred = model(image).cpu().detach()
                preds.append(pred)
            preds = torch.cat(preds, dim=0)

            preds = (
                preds.argmax(dim=1)
                if preds.shape[1] > 1
                else (preds > 0).long().squeeze(1)
            )

            if dims == 5:
                preds = preds.reshape(B, D, H, W)

            num_samples += B
            for i in range(B):
                case_metrics = metrics_fn(preds[i], labels[i])
                for k, v in case_metrics.items():
                    metrics[k] += v

    for k in metrics:
        metrics[k] /= num_samples

    for k, v in metrics.items():
        writer.add_scalar(f"metrics/{k}", v, epoch)

    model.train()
    return metrics[primary_metric]


def train_eval(
    args,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_obj_dict: dict,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    metrics_fn,
    primary_metric: str,
):

    model.train()

    writer = SummaryWriter(os.path.join(args.outputs_dir, "log"))

    if os.path.exists(args.checkpoint_fpath):
        logging.info("Training restart...")
        checkpoint = torch.load(args.checkpoint_fpath, weights_only=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        epoch = checkpoint["epoch"]
        best_performance = checkpoint["best_performance"]

        ret = load_model_state_dict(model, checkpoint["model_state_dict"])
        msg = [f"Loaded model weights from {args.checkpoint_fpath}"]
        if ret.missing_keys:
            msg.append(f"Missing keys: {ret.missing_keys}")
        if ret.unexpected_keys:
            msg.append(f"Unexpected keys: {ret.unexpected_keys}")

        logging.info("\n".join(msg))

    else:
        iteration = 0
        epoch = 0
        best_performance = 0

    while epoch < args.max_epochs:
        epoch += 1

        for batch in train_loader:
            iteration += 1
            images = batch["image"].cuda()
            labels = batch["label"].cuda()
            bnd_labels = batch["bnd_label"].cuda()

            preds, gs = model(images)

            loss_total = 0.0
            # main loss
            for loss_obj in loss_obj_dict["main"]:
                if not isinstance(preds, (tuple, list)):
                    preds = [preds]

                loss = sum(
                    [
                        loss_obj["calc"](pred, labels) * loss_obj["weight"]
                        for pred in preds
                    ]
                ) / len(preds)
                writer.add_scalar(f"loss/{loss_obj['name']}", loss, iteration)
                loss_total += loss

            # bnd loss
            for loss_obj in loss_obj_dict["bnd"]:
                loss = sum(
                    [
                        loss_obj["calc"](g, bnd_labels)
                        * loss_obj["weight"]
                        for g in gs
                    ]
                )
                writer.add_scalar(f"loss/{loss_obj['name']}", loss, iteration)
                loss_total += loss

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar("loss/total_loss", loss_total, iteration)

            # visualization
            if iteration % 20 == 0:
                image = images[0].cpu().detach()
                image = (image - image.min()) / (image.max() - image.min() + 1e-5)
                writer.add_image("visual/image", image, iteration)

                pred = preds[0][0].cpu().detach()
                pred = (
                    pred.argmax(dim=0, keepdim=True).float()
                    if pred.shape[0] > 1
                    else (pred > 0).float()
                )
                pred = pred / args.num_classes
                writer.add_image("visual/pred", pred, iteration)

                g = gs[-1][0].cpu().detach()
                g = (g - g.min()) / (g.max() - g.min() + 1e-5)
                writer.add_image("visual/bnd_pred", g, iteration)

                label = labels[0].cpu().detach().unsqueeze(0).float()
                label = label / args.num_classes
                writer.add_image("visual/label", label, iteration)

                bnd_label = bnd_labels[0].cpu().detach().unsqueeze(0)
                writer.add_image("visual/bnd_label", bnd_label, iteration)

        if epoch > args.warmup_epochs:
            performance = eval(
                args,
                model,
                val_loader,
                epoch,
                writer,
                metrics_fn,
                primary_metric,
            )

            if performance > best_performance:
                best_performance = performance
                tmp_path = args.best_model_fpath + ".tmp"
                torch.save(get_model_state_dict(model), tmp_path)
                os.replace(tmp_path, args.best_model_fpath)
                logging.info(
                    f"epoch {epoch:<3d}\tlr: {optimizer.param_groups[0]['lr']:.6f}\t{primary_metric}: {best_performance:.6f} *"
                )

            else:
                logging.info(
                    f"epoch {epoch:<3d}\tlr: {optimizer.param_groups[0]['lr']:.6f}\t{primary_metric}: {performance:.6f}"
                )

            if epoch % args.checkpoint_save_interval == 0:
                checkpoint = {
                    "model_state_dict": get_model_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "iteration": iteration,
                    "epoch": epoch,
                    "best_performance": best_performance,
                }

                tmp_path = args.checkpoint_fpath + ".tmp"
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, args.checkpoint_fpath)

        else:
            logging.info(
                f"epoch {epoch:<3d}\tlr: {optimizer.param_groups[0]['lr']:.6f}\twarmup: {epoch}/{args.warmup_epochs}"
            )

    logging.info("Training completed!")
    logging.info(f"Best {primary_metric} -----> {best_performance:.6f}")


def train(
    args,
    model: nn.Module,
    train_loader: DataLoader,
    loss_obj_dict: dict,
    optimizer: Optimizer,
    scheduler: LRScheduler,
):

    model.train()

    writer = SummaryWriter(os.path.join(args.outputs_dir, "log"))

    if os.path.exists(args.checkpoint_fpath):
        logging.info("Training restart...")
        checkpoint = torch.load(args.checkpoint_fpath, weights_only=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        epoch = checkpoint["epoch"]

        ret = load_model_state_dict(model, checkpoint["model_state_dict"])
        msg = [f"Loaded model weights from {args.checkpoint_fpath}"]
        if ret.missing_keys:
            msg.append(f"Missing keys: {ret.missing_keys}")
        if ret.unexpected_keys:
            msg.append(f"Unexpected keys: {ret.unexpected_keys}")

        logging.info("\n".join(msg))

    else:
        iteration = 0
        epoch = 0

    while epoch < args.max_epochs:
        epoch += 1

        for batch in train_loader:
            iteration += 1
            images = batch["image"].cuda()
            labels = batch["label"].cuda()
            bnd_labels = batch["bnd_label"].cuda()

            preds, gs = model(images)

            loss_total = 0.0
            # main loss
            for loss_obj in loss_obj_dict["main"]:
                if not isinstance(preds, (tuple, list)):
                    preds = [preds]

                loss = sum(
                    [
                        loss_obj["calc"](pred, labels) * loss_obj["weight"]
                        for pred in preds
                    ]
                ) / len(preds)
                writer.add_scalar(f"loss/{loss_obj['name']}", loss, iteration)
                loss_total += loss

            # bnd loss
            for loss_obj in loss_obj_dict["bnd"]:
                loss = sum(
                    [
                        loss_obj["calc"](g, bnd_labels)
                        * loss_obj["weight"]
                        for g in gs
                    ]
                )
                writer.add_scalar(f"loss/{loss_obj['name']}", loss, iteration)
                loss_total += loss

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar("loss/total_loss", loss_total, iteration)

            # visualization
            if iteration % 20 == 0:
                image = images[0].cpu().detach()
                image = (image - image.min()) / (image.max() - image.min() + 1e-5)
                writer.add_image("visual/image", image, iteration)

                pred = preds[0][0].cpu().detach()
                pred = (
                    pred.argmax(dim=0, keepdim=True).float()
                    if pred.shape[0] > 1
                    else (pred > 0).float()
                )
                pred = pred / args.num_classes
                writer.add_image("visual/pred", pred, iteration)

                g = gs[-1][0].cpu().detach()
                g = (g - g.min()) / (g.max() - g.min() + 1e-5)
                writer.add_image("visual/bnd_pred", g, iteration)

                label = labels[0].cpu().detach().unsqueeze(0).float()
                label = label / args.num_classes
                writer.add_image("visual/label", label, iteration)

                bnd_label = bnd_labels[0].cpu().detach().unsqueeze(0)
                writer.add_image("visual/bnd_label", bnd_label, iteration)

        if epoch % args.checkpoint_save_interval == 0:
            checkpoint = {
                "model_state_dict": get_model_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "iteration": iteration,
                "epoch": epoch,
            }

            tmp_path = args.checkpoint_fpath + ".tmp"
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, args.checkpoint_fpath)

        if epoch > args.warmup_epochs:
            logging.info(
                f"epoch {epoch:<3d}\tlr: {optimizer.param_groups[0]['lr']:.6f}"
            )

        else:
            logging.info(
                f"epoch {epoch:<3d}\tlr: {optimizer.param_groups[0]['lr']:.6f}\twarmup: {epoch}/{args.warmup_epochs}"
            )

    torch.save(get_model_state_dict(model), args.best_model_fpath)
    logging.info("Training completed!")


def train_polypdb_wli(args):

    model = get_model(args)

    train_db = PolypDB_WLI(root=args.root, split="train")
    val_db = PolypDB_WLI(root=args.root, split="valid")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(
        val_db,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    metrics_fn = metrics_binary
    primary_metric = "dice"

    train_eval(
        args,
        model,
        train_loader,
        val_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
        metrics_fn,
        primary_metric,
    )


def train_bkai(args):

    model = get_model(args)

    train_db = BKAI(root=args.root, split="train")
    val_db = BKAI(root=args.root, split="valid")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(
        val_db,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    metrics_fn = metrics_binary
    primary_metric = "dice"

    train_eval(
        args,
        model,
        train_loader,
        val_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
        metrics_fn,
        primary_metric,
    )


def train_busi(args):

    model = get_model(args)

    train_db = BUSI(root=args.root, split="train")
    val_db = BUSI(root=args.root, split="valid")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(
        val_db,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    metrics_fn = metrics_binary
    primary_metric = "dice"

    train_eval(
        args,
        model,
        train_loader,
        val_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
        metrics_fn,
        primary_metric,
    )


def train_busbra(args):

    model = get_model(args)

    train_db = BUSBRA(root=args.root, split="train")
    val_db = BUSBRA(root=args.root, split="valid")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(
        val_db,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    metrics_fn = metrics_binary
    primary_metric = "dice"

    train_eval(
        args,
        model,
        train_loader,
        val_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
        metrics_fn,
        primary_metric,
    )


def train_bus_uc(args):

    model = get_model(args)

    train_db = BUS_UC(root=args.root, split="train")
    val_db = BUS_UC(root=args.root, split="valid")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(
        val_db,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    metrics_fn = metrics_binary
    primary_metric = "dice"

    train_eval(
        args,
        model,
        train_loader,
        val_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
        metrics_fn,
        primary_metric,
    )


def train_brisc2025(args):

    model = get_model(args)

    train_db = BRISC2025(root=args.root, split="train")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    train(
        args,
        model,
        train_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
    )


def train_kvasirseg(args):

    model = get_model(args)

    train_db = KvasirSEG(root=args.root, split="train")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    train(
        args,
        model,
        train_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
    )


def train_isic2017(args):

    model = get_model(args)

    train_db = ISIC2017(root=args.root, split="train")
    val_db = ISIC2017(root=args.root, split="valid")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(
        val_db,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    metrics_fn = metrics_binary
    primary_metric = "dice"

    train_eval(
        args,
        model,
        train_loader,
        val_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
        metrics_fn,
        primary_metric,
    )


def train_isic2018(args):

    model = get_model(args)

    train_db = ISIC2018(root=args.root, split="train")
    val_db = ISIC2018(root=args.root, split="valid")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(
        val_db,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    metrics_fn = metrics_binary
    primary_metric = "dice"

    train_eval(
        args,
        model,
        train_loader,
        val_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
        metrics_fn,
        primary_metric,
    )


def train_chaos(args):
    case_ids = sorted(os.listdir(os.path.join(args.root, "slices", "images")))
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=1234)
    outputs_dir = args.outputs_dir

    for i, (train_ids, valid_ids) in enumerate(kf.split(case_ids)):
        train_case_ids = [case_ids[j] for j in train_ids]
        valid_case_ids = [case_ids[j] for j in valid_ids]

        args.outputs_dir = os.path.join(outputs_dir, f"fold_{i}")
        if os.path.exists(args.outputs_dir):
            continue

        os.makedirs(args.outputs_dir, exist_ok=True)

        args.checkpoint_fpath = os.path.join(args.outputs_dir, "checkpoint.pth")
        args.best_model_fpath = os.path.join(args.outputs_dir, "best_model.pth")

        log_fpath = os.path.join(args.outputs_dir, f"train_log.txt")
        logging.basicConfig(
            filename=log_fpath,
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        model = get_model(args)

        train_db = CHAOS(root=args.root, case_ids=train_case_ids, split="train")
        val_db = CHAOS(root=args.root, case_ids=valid_case_ids, split="valid")
        train_loader = DataLoader(
            train_db,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
        )
        val_loader = DataLoader(
            val_db,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        loss_obj_dict = {
            "main": [
                {
                    "name": "main_bce_loss",
                    "calc": BCEWithLogitsLoss(),
                    "weight": 0.5,
                },
                {
                    "name": "main_dice_loss",
                    "calc": DiceLoss(),
                    "weight": 0.5,
                },
            ],
            "bnd": [
                {
                    "name": "bnd_bce_loss",
                    "calc": BCELoss(),
                    "weight": args.bnd_weight / 2,
                },
                {
                    "name": "bnd_dice_loss",
                    "calc": DiceLoss(),
                    "weight": args.bnd_weight / 2,
                },
            ],
        }

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay,
        )

        iters_per_epoch = len(train_loader)
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_iters=args.warmup_epochs * iters_per_epoch,
            max_iters=args.max_epochs * iters_per_epoch,
        )

        metrics_fn = metrics_binary
        primary_metric = "dice"

        train_eval(
            args,
            model,
            train_loader,
            val_loader,
            loss_obj_dict,
            optimizer,
            scheduler,
            metrics_fn,
            primary_metric,
        )


def train_qata_cov19_v2(args):

    model = get_model(args)

    train_db = QaTa_COV19_v2(root=args.root, split="train")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    train(
        args,
        model,
        train_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
    )


def train_qata_cov19_v1(args):

    model = get_model(args)

    train_db = QaTa_COV19_v1(root=args.root, split="train")
    val_db = QaTa_COV19_v1(root=args.root, split="valid")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(
        val_db,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    metrics_fn = metrics_binary
    primary_metric = "dice"

    train_eval(
        args,
        model,
        train_loader,
        val_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
        metrics_fn,
        primary_metric,
    )
    

def train_btmri(args):

    model = get_model(args)

    train_db = BTMRI(root=args.root, split="train")
    val_db = BTMRI(root=args.root, split="valid")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(
        val_db,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    metrics_fn = metrics_binary
    primary_metric = "dice"

    train_eval(
        args,
        model,
        train_loader,
        val_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
        metrics_fn,
        primary_metric,
    )


def train_nusec(args):

    model = get_model(args)

    train_db = NuSeC(root=args.root, split="train")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    train(
        args,
        model,
        train_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
    )


def train_glas(args):

    model = get_model(args)

    train_db = GlaS(root=args.root, split="train")
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )

    loss_obj_dict = {
        "main": [
            {
                "name": "main_bce_loss",
                "calc": BCEWithLogitsLoss(),
                "weight": 0.5,
            },
            {
                "name": "main_dice_loss",
                "calc": DiceLoss(),
                "weight": 0.5,
            },
        ],
        "bnd": [
            {
                "name": "bnd_bce_loss",
                "calc": BCELoss(),
                "weight": args.bnd_weight / 2,
            },
            {
                "name": "bnd_dice_loss",
                "calc": DiceLoss(),
                "weight": args.bnd_weight / 2,
            },
        ],
    }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    train(
        args,
        model,
        train_loader,
        loss_obj_dict,
        optimizer,
        scheduler,
    )