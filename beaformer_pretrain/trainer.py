import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import os
from tensorboardX import SummaryWriter
import logging
from utils import *
from datasets.COD27K import COD27K
import random
from scheduler import WarmupCosineAnnealingLR
from loss import *


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
            dist_weights = None
            if args.use_dist_weight:
                dist_weights = batch["dist_weight"].cuda()

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
                        loss_obj["calc"](g, bnd_labels, dist_weights)
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
            logging.info(
                f"epoch {epoch:<3d}\tlr: {optimizer.param_groups[0]['lr']:.6f}"
            )

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

        else:
            logging.info(
                f"epoch {epoch:<3d}\tlr: {optimizer.param_groups[0]['lr']:.6f}\twarmup: {epoch}/{args.warmup_epochs}"
            )

    model_state = get_model_state_dict(model)
    torch.save(model_state, os.path.join(args.outputs_dir, "model_state.pth"))
    model_state = {
        k.split(".", 1)[1]: v
        for k, v in model_state.items()
        if not k.startswith("decoder")
    }
    torch.save(model_state, os.path.join(args.outputs_dir, "backbone.pth"))
    logging.info("Training completed!")


def train_cod27k(args):

    model = get_model(args)

    train_db = COD27K(root=args.root, use_dist_weight=args.use_dist_weight)
    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
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
        params=model.freeze() if args.freeze else model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
        max_iters=args.max_epochs * iters_per_epoch,
    )

    train(args, model, train_loader, loss_obj_dict, optimizer, scheduler)
