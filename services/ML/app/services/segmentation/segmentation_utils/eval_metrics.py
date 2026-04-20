import csv
import os

import matplotlib
import numpy as np
import torch

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

##########################
### METRIC COMPUTATION ###


# Compute dice coefficient for binary segmentation (Unet training and testing)
def compute_dice(y_true, y_pred):
    """
    Args:
        y_true: Ground truth masks (numpy array or tensor, any dimension)
        y_pred: Predicted masks (same shape as y_true)
    Returns:
        float: Dice coefficient for foreground class (class=1)
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)

    # Compute foreground-only Dice coefficient (class == 1)
    intersection = ((y_pred == 1) & (y_true == 1)).sum()
    total = (y_pred == 1).sum() + (y_true == 1).sum()
    return (2.0 * intersection) / (total + 1e-8) if total > 0 else 0.0


# Compute IoU for binary segmentation (between predicted and ground truth masks)
def compute_iou(y_true, y_pred):
    """
    Args:
        y_true: Ground truth masks (numpy array or tensor, any dimension)
        y_pred: Predicted masks (same shape as y_true)
    Returns:
        float: IoU for foreground class (class=1)
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)

    # Compute foreground-only IoU (class == 1)
    pred_fg = y_pred == 1
    true_fg = y_true == 1
    intersection = (pred_fg & true_fg).sum()
    union = (pred_fg | true_fg).sum()
    return intersection / (union + 1e-8) if union > 0 else 0.0


#######################################
### VALIDATION EVALUATION FUNCTIONS ###


# --- Evaluation function for U-Net variants and DeepLabV3+ ---
def evaluate_semantic_model(model, data_loader, device, criterion):
    """
    Evaluate U-Net variants or DeepLabV3+ model during validation in training.
    Computes per-image (patch) IoU and Dice, then averages across images (patches in data_loader).
    Returns: (mean_loss, mean_iou, mean_dice)
    """
    model.eval()
    total_loss = 0.0
    ious = []
    dices = []
    total_images = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)
            # loss.item() is averaged over all elements in the batch (N x H x W)
            # convert to per-image aggregated loss so we can average per-image later
            total_loss += loss.item() * images.size(0)

            # compute per-image metrics
            for i in range(images.size(0)):
                gt_i = masks[i].cpu()
                pred_i = preds[i].cpu()
                iou_i = compute_iou(gt_i, pred_i)
                dice_i = compute_dice(gt_i, pred_i)
                ious.append(iou_i)
                dices.append(dice_i)

            total_images += images.size(0)

    # Average metrics of all patch images (during validation)
    mean_loss = (total_loss / total_images) if total_images > 0 else 0.0
    mean_iou = float(sum(ious) / len(ious)) if ious else 0.0
    mean_dice = float(sum(dices) / len(dices)) if dices else 0.0
    return mean_loss, mean_iou, mean_dice


# --- Evaluation function for Mask R-CNN ---
def evaluate_maskrcnn(model, data_loader, device, criterion, confidence):
    """
    Compute mean IoU and Dice for a Mask R-CNN model (during training) by converting instance masks
    into a single prediction mask per image patch (union of predicted instance masks).
    The confidence is used as the probability threshold to binarize the per-pixel predicted union (`pred_prob_union`).
    This function returns the mean detection loss (the same scalar used during training) to compare training/validation losses.
    """
    model.eval()
    ious = []
    dices = []
    # detection loss (as returned by Mask R-CNN when called with targets)
    detection_total_loss = 0.0
    detection_total_images = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images_list = [img.to(device) for img in images]

            # compute detection losses (patch-wise) by calling model with targets
            try:
                targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # switch to train mode for loss computation
                model.train()
                loss_dict = model(images_list, targets_dev)
                # sum all detection loss terms (loss_dict values are typically averaged per-image by the model)
                loss_val = float(sum([v.item() for v in loss_dict.values()])) if loss_dict else 0.0
                batch_size = len(images_list)
                # accumulate weighted by number of images in the batch to compute per-sample mean
                detection_total_loss += loss_val * batch_size
                detection_total_images += batch_size
            finally:
                # restore eval mode
                model.eval()

            outputs = model(images_list)

            # iterate per image
            for out, tgt in zip(outputs, targets):
                # predicted union mask (probabilities)
                if "masks" in out and out["masks"] is not None and out["masks"].numel() > 0:
                    # out['masks']: [N_instances, 1, H, W]
                    masks_pred = out["masks"].squeeze(1).to(device)  # [N, H, W]
                    # per-pixel probability of being foreground = max over instance probabilities
                    pred_prob_union = masks_pred.max(dim=0).values  # [H, W], on device
                else:  # no predicted masks
                    # infer H,W from image tensor
                    H = images_list[0].shape[1] if images_list else 0
                    W = images_list[0].shape[2] if images_list else 0
                    pred_prob_union = torch.zeros((H, W), device=device)

                # ground truth union (0/1) on device
                if "masks" in tgt and tgt["masks"] is not None and tgt["masks"].numel() > 0:
                    gt_masks = tgt["masks"].to(device)  # [N_gt, H, W]
                    gt_union = (gt_masks > 0).any(dim=0).to(torch.long)
                else:
                    H = pred_prob_union.shape[0]
                    W = pred_prob_union.shape[1]
                    gt_union = torch.zeros((H, W), dtype=torch.long, device=device)

                # compute binary predicted union -> Threshold probability map at confidence to obtain predicted union mask (0/1)
                pred_union = (pred_prob_union > float(confidence)).to(torch.long).cpu()
                # compute foreground-only IoU/Dice
                iou_val = compute_iou(gt_union.cpu(), pred_union)
                dice_val = compute_dice(gt_union.cpu(), pred_union)
                ious.append(iou_val)
                dices.append(dice_val)

    # Average metrics of all patch images (during validation)
    mean_iou = float(sum(ious) / len(ious)) if ious else 0.0
    mean_dice = float(sum(dices) / len(dices)) if dices else 0.0
    mean_detection_loss = (detection_total_loss / detection_total_images) if detection_total_images > 0 else 0.0

    return mean_detection_loss, mean_iou, mean_dice


# --- Log evaluation metrics for later inspection (.csv) ---
def log_evaluation_metrics(
    log_file_path,
    epoch,
    train_loss,
    val_loss,
    iou,
    dice,
    lr=None,
    best_iou=None,
    epochs_without_improvement=None,
    epoch_time=None,
):
    """
    Args:
        log_file_path (str): path to CSV file to append
        epoch (int): 1-based epoch index
        train_loss (float): training loss for the epoch
        val_loss (float): validation loss for the epoch
        iou (float): validation IoU for the epoch
        dice (float): validation Dice for the epoch
        lr (float, optional): current learning rate (for current epoch)
        best_iou (float, optional): best IoU seen in training so far
        epochs_without_improvement (int, optional): count of epochs since improvement
        epoch_time (float, optional): time taken for the epoch in seconds
    """
    # Ensure directory exists, if not, create it
    parent = os.path.dirname(log_file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    file_exists = os.path.exists(log_file_path)
    # Append metrics to CSV file (or create with header if not exists)
    with open(log_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_iou",
                    "val_dice",
                    "lr",
                    "best_iou",
                    "epochs_without_improvement",
                    "epoch_time",
                ]
            )
        row = [
            int(epoch),
            f"{float(train_loss):.6f}",
            f"{float(val_loss):.6f}",
            f"{float(iou):.6f}",
            f"{float(dice):.6f}",
            f"{float(lr):.8e}" if lr is not None else "",
            f"{float(best_iou):.6f}" if best_iou is not None else "",
            int(epochs_without_improvement) if epochs_without_improvement is not None else "",
            f"{float(epoch_time):.2f}" if epoch_time is not None else "",
        ]
        writer.writerow(row)


# Read a training CSV produced by `log_evaluation_metrics`
def plot_training_curves(log_file_path, save_dir=None):
    """
    Generates three plots:
        - train/val loss vs epoch (saved as `loss_curve.png`)
        - val IoU and Dice vs epoch (saved as `metrics_curve.png`)
        - learning rate vs epoch with vertical lines where LR changes (saved as `lr_curve.png`)

    Args:
        log_file_path (str): path to CSV file written by `log_evaluation_metrics`.
        save_dir (str, optional): directory to save plots. If None, the CSV parent directory is used.
    Returns:
        dict: paths of saved plot files (keys: loss, metrics, lr)
    """
    if not os.path.exists(log_file_path):
        print(f"Warning: log file not found: {log_file_path}")
        return {}

    parent = os.path.dirname(log_file_path)
    out_dir = save_dir if save_dir is not None else parent
    os.makedirs(out_dir, exist_ok=True)
    # Read CSV
    epochs = []
    train_loss = []
    val_loss = []
    val_iou = []
    val_dice = []
    lr_list = []
    best_iou_list = []

    with open(log_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row.get("epoch", "")))
            except Exception:
                epochs.append(None)
            try:
                train_loss.append(float(row.get("train_loss", "") or np.nan))
            except Exception:
                train_loss.append(np.nan)
            try:
                val_loss.append(float(row.get("val_loss", "") or np.nan))
            except Exception:
                val_loss.append(np.nan)
            try:
                val_iou.append(float(row.get("val_iou", "") or np.nan))
            except Exception:
                val_iou.append(np.nan)
            try:
                val_dice.append(float(row.get("val_dice", "") or np.nan))
            except Exception:
                val_dice.append(np.nan)
            try:
                lr_list.append(float(row.get("lr", "") or np.nan))
            except Exception:
                lr_list.append(np.nan)
            try:
                best_iou_list.append(float(row.get("best_iou", "") or np.nan))
            except Exception:
                best_iou_list.append(np.nan)

    # Convert to numpy arrays for convenience
    epochs_arr = (
        np.array([e for e in epochs if e is not None])
        if any(e is not None for e in epochs)
        else np.arange(1, len(train_loss) + 1)
    )
    x = epochs_arr if len(epochs_arr) == len(train_loss) else np.arange(1, len(train_loss) + 1)

    # Best epoch by val_iou
    best_epoch = None
    try:
        vi = np.array(val_iou, dtype=float)
        if np.any(~np.isnan(vi)):
            best_idx = int(np.nanargmax(vi))
            best_epoch = int(x[best_idx])
    except Exception:
        best_epoch = None

    saved = {}

    # Plot Losses
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(x, train_loss, label="train_loss", marker="o")
        plt.plot(x, val_loss, label="val_loss", marker="o")
        if best_epoch is not None:
            plt.axvline(best_epoch, color="green", linestyle="--", label=f"best_epoch={best_epoch}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        loss_path = os.path.join(out_dir, "loss_curve.png")
        plt.tight_layout()
        plt.savefig(loss_path)
        plt.close()
        saved["loss"] = loss_path
    except Exception as e:
        print(f"Warning: could not plot loss curve: {e}")

    # Plot IoU and Dice
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(x, val_iou, label="val_IoU", marker="o")
        plt.plot(x, val_dice, label="val_Dice", marker="o")
        if best_epoch is not None:
            plt.axvline(best_epoch, color="green", linestyle="--", label=f"best_epoch={best_epoch}")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Validation IoU and Dice")
        plt.legend()
        plt.grid(True)
        metrics_path = os.path.join(out_dir, "metrics_curve.png")
        plt.tight_layout()
        plt.savefig(metrics_path)
        plt.close()
        saved["metrics"] = metrics_path
    except Exception as e:
        print(f"Warning: could not plot metrics curve: {e}")

    print(f"Saved plots: {saved}")
    return saved
