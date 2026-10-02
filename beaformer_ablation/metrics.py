import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from medpy import metric
import math


def metrics_binary(preds: torch.Tensor, labels: torch.Tensor):
    y_pred = preds.cpu().detach().numpy().astype(np.uint8)
    y_true = labels.cpu().detach().numpy().astype(np.uint8)

    metrics = {}

    if np.any(y_pred) and np.any(y_true):
        metrics["dice"] = metric.binary.dc(y_pred, y_true)
        metrics["iou"] = metric.binary.jc(y_pred, y_true)
        metrics["hd95"] = metric.binary.hd95(y_pred, y_true)
        metrics["assd"] = metric.binary.assd(y_pred, y_true)

    elif not np.any(y_pred) and not np.any(y_true):
        metrics["dice"] = 1
        metrics["iou"] = 1
        metrics["hd95"] = 0
        metrics["assd"] = 0

    else:
        metrics["dice"] = 0
        metrics["iou"] = 0
        metrics["hd95"] = 224 * math.sqrt(2)  # diagonal
        metrics["assd"] = 224 * math.sqrt(2)

    return metrics


def metrics_multi(preds: torch.Tensor, labels: torch.Tensor, num_classes: int, skip_background: bool = True):
    y_pred = preds.cpu().detach().numpy()
    y_true = labels.cpu().detach().numpy()

    metrics = {}
    dice = []
    hd95 = []
    iou = []
    assd = []
    for i in range(1 if skip_background else 0, num_classes):
        y_pred_i = (y_pred == i).astype(np.uint8)
        y_true_i = (y_true == i).astype(np.uint8)

        if np.any(y_pred_i) and np.any(y_true_i):
            dice_i = metric.binary.dc(y_pred_i, y_true_i)
            iou_i = metric.binary.jc(y_pred_i, y_true_i)
            hd95_i = metric.binary.hd95(y_pred_i, y_true_i)
            assd_i = metric.binary.assd(y_pred_i, y_true_i)

        elif not np.any(y_pred_i) and not np.any(y_true_i):
            dice_i = 1
            iou_i = 1
            hd95_i = 0
            assd_i = 0

        else:
            dice_i = 0
            iou_i = 0
            hd95_i = 224 * math.sqrt(2)
            assd_i = 224 * math.sqrt(2)

        dice.append(dice_i)
        iou.append(iou_i)
        hd95.append(hd95_i)
        assd.append(assd_i)

        metrics[f"class_{i}_dice"] = dice_i
        metrics[f"class_{i}_iou"] = iou_i
        metrics[f"class_{i}_hd95"] = hd95_i
        metrics[f"class_{i}_assd"] = assd_i

    metrics["mean_dice"] = np.mean(dice)
    metrics["mean_iou"] = np.mean(iou)
    metrics["mean_hd95"] = np.mean(hd95)
    metrics["mean_assd"] = np.mean(assd)

    return metrics

