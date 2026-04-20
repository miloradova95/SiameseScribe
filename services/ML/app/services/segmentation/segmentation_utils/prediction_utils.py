import warnings

warnings.filterwarnings("ignore")

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from patchify import patchify
from PIL import Image
from segmentation_utils.eval_metrics import compute_dice, compute_iou

### FULL IMAGE PREDICTION FUNCTION ###


# Predict a full-image mask from a single RGB image using a patch-based sliding window
def get_image_prediction(
    original_image_rgb, patch_size, step, device, model, confidence_thresh=None, model_type="semantic"
):
    """
    Inputs:
      - original_image_rgb: numpy array HxWx3 (uint8)
      - patch_size: int
      - step: int (sliding step for patchify)
      - device: torch.device
      - model: loaded model (U-Net variant, DeepLabV3+, or Mask R-CNN)
      - confidence_thresh: float for maskrcnn (ignored for semantic)
      - model_type: 'semantic' or 'maskrcnn'

    Returns: (binary_mask, raw_prediction)
      - binary_mask: HxW uint8 array with values {0,1}
      - raw_prediction: HxW array (integer counts for patch votes, or None if unavailable)
      - gray: HxW uint8 array (grayscale mask for semantic models, or None)

    Notes:
      - Black borders added for patching are removed from the predictions (binary, raw and gray map) before returning
      - Blob removal & saving are intentionally left as a separate step
    """
    # Validate input
    if original_image_rgb is None:
        raise ValueError("original_image_rgb must be provided")
    if original_image_rgb.ndim != 3 or original_image_rgb.shape[2] != 3:
        raise ValueError("original_image_rgb must be HxWx3 RGB image")

    # Add black borders so image dimensions fit sliding window
    img_bordered = return_image_black_borders(original_image_rgb, patch_size, step)
    # Create patches with a sliding window
    patches = patchify(img_bordered, (patch_size, patch_size, 3), step)

    # RUN RECONSTRUCTION (depending on model type)
    if model_type == "maskrcnn":
        # --- Instance segmentation model (Mask R-CNN) ---
        if confidence_thresh is None:
            raise ValueError("confidence_thresh must be provided for maskrcnn model_type")
        storage = reconstruction_maskrcnn(
            patches,
            device=device,
            model=model,
            orig_size=img_bordered.shape,
            step=step,
            confidence_thresh=confidence_thresh,
        )

        # Crop padded regions
        storage_cropped, _, _ = remove_black_borders(storage, og_mask_image=None, og_rgb_image=original_image_rgb)

        # Extract binary and raw
        try:
            binary = storage_cropped.output_array[0]
        except Exception:
            # fallback: binarize raw if available
            binary = (
                (storage_cropped.raw_reconstructed != 0).astype(np.uint8)
                if (hasattr(storage_cropped, "raw_reconstructed") and storage_cropped.raw_reconstructed is not None)
                else None
            )
        raw = storage_cropped.raw_reconstructed if hasattr(storage_cropped, "raw_reconstructed") else None
        gray = None  # No gray map for Mask R-CNN

    else:
        # --- Semantic segmentation models (UNet, DeepLabV3+) ---
        binary_arr, raw_counts, mask_gray = reconstruction_semantic(
            patches, device=device, model=model, orig_size=img_bordered.shape, step=step
        )
        # Crop padded regions (remove_black_borders accepts tuple)
        cropped, _, _ = remove_black_borders(
            (binary_arr, raw_counts, mask_gray), og_mask_image=None, og_rgb_image=original_image_rgb
        )
        try:
            # set binary, raw, and gray mask from cropped tuple
            binary, raw, gray = cropped
        except Exception:
            binary = cropped
            raw = None
            gray = None

    # Ensure binary is uint8 0/1
    if binary is None:
        raise RuntimeError("Failed to obtain binary prediction")
    binary = (binary > 0).astype(np.uint8)

    return binary, raw, gray


####################################################
### SEMANTIC RECONSTRUCTION (U-Net + DeepLabV3+) ###


# Reconstruct a full-size prediction from semantic segmentation patch predictions
def reconstruction_semantic(patches_array, device, model, orig_size, step):
    """
    Returns: (binary_mask, reconstructed_counts, avg_prob_map)
    - binary_mask: uint8 array of {0,1}
    - reconstructed_counts: integer array showing how many times a pixel was predicted positive (raw before binarization)
    - avg_prob_map: float32 array of per-pixel averaged probabilities for the positive class (in [0,1])
        -> This can be used as a gray map to visualize prediction confidence for semantic models
    """
    num_patches_0 = patches_array.shape[0]
    num_patches_1 = patches_array.shape[1]
    patch_size = patches_array.shape[3]
    reconstructed_counts = np.zeros(orig_size[:2], dtype=np.uint16)
    # Accumulate per-pixel probability sums for the positive (foreground) class
    reconstructed_prob_sum = np.zeros(orig_size[:2], dtype=np.float32)
    # Count how many patch contributions each pixel receives (for averaging probs)
    prob_contrib_counts = np.zeros(orig_size[:2], dtype=np.uint16)

    # Iterate over each patch and predict mask
    for i in range(num_patches_0):
        for j in range(num_patches_1):
            patch = patches_array[i, j, 0]
            patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                output = model(patch_tensor)
                # Output shape expected: (1, num_classes, H, W)
                probs = torch.softmax(output, dim=1)
                # predicted class per pixel (0..num_classes-1)
                mask_pred = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
                # per-pixel foreground probability (assume class 1 is foreground)
                try:
                    probs_pos = probs[0, 1].cpu().numpy()
                except Exception:
                    # If model outputs single-channel logits, apply sigmoid instead
                    probs_pos = torch.sigmoid(output.squeeze(0)[0]).cpu().numpy()

            # Place patch mask votes in reconstructed counts
            row_start = i * step
            row_end = row_start + patch_size
            col_start = j * step
            col_end = col_start + patch_size
            reconstructed_counts[row_start:row_end, col_start:col_end] += (mask_pred == 1).astype(np.uint16)

            # Accumulate probabilities and contribution counts for averaging later
            reconstructed_prob_sum[row_start:row_end, col_start:col_end] += probs_pos
            prob_contrib_counts[row_start:row_end, col_start:col_end] += 1

    # Create binary mask from counts (any positive vote -> 1)
    reconstructed_binary = (reconstructed_counts != 0).astype(np.uint8)

    # Compute averaged probability map where at least one patch contributed
    avg_prob_map = np.zeros_like(reconstructed_prob_sum, dtype=np.float32)
    contrib_mask = prob_contrib_counts > 0
    avg_prob_map[contrib_mask] = reconstructed_prob_sum[contrib_mask] / prob_contrib_counts[contrib_mask]

    # Note: avg_prob_map is in [0,1] and can be returned for visualization (gray_map)
    return reconstructed_binary, reconstructed_counts, avg_prob_map


#################################
### MASK R-CNN RECONSTRUCTION ###


# Storage class - initializes storage for patch-wise mask predictions and confidence scores
class MaskConf_storage:
    def __init__(self):
        """
        Attributes:
            arrays_to_unpatch (np.ndarray): Holds predicted patch masks for reconstruction
            confidence_scores (np.ndarray): Average confidence score for each patch
            output_array (np.ndarray): Final reconstructed mask(s) for the image(s)
            output_confidence_score (float): Mean confidence score across all patches for the image
            raw_reconstructed (np.ndarray): Raw reconstructed counts for the image (before binarization)
        """
        self.arrays_to_unpatch = (
            None  # shape: (total_patches, num_patches_0, num_patches_1, 1, patch_size, patch_size, 1)
        )
        self.confidence_scores = None  # shape: (total_patches,)
        self.output_array = None  # shape: (num_images, H, W)
        self.output_confidence_score = 0
        self.raw_reconstructed = None


# Batch size for how many patches to process at once during Mask R-CNN reconstruction
batch_size = 8


# Main function for reconstructing a full-size mask from predicted patches
def reconstruction_maskrcnn(patches_array, device, model, orig_size, step, confidence_thresh, batch_size=batch_size):
    # Convert image patches to tensors for model input
    transform = T.Compose([T.ToTensor()])

    # Initialize storage
    storage = MaskConf_storage()
    divide_sum = 0
    num_patches_0 = patches_array.shape[0]  # Number of patches (height)
    num_patches_1 = patches_array.shape[1]  # Number of patches (width)
    total_patches = num_patches_0 * num_patches_1

    # Pre-allocate storage for all patch predictions and confidences
    # Only one patch is processed at a time, so we only need to store the current patch's mask and confidence
    patch_size = patches_array.shape[3]
    storage.arrays_to_unpatch = np.zeros(
        (total_patches, num_patches_0, num_patches_1, 1, patch_size, patch_size, 1), dtype=np.uint8
    )
    storage.confidence_scores = np.zeros((total_patches,), dtype=np.float32)
    e = 0  # Index for storage arrays

    # Final reconstructed mask (counts before binarization)
    reconstructed_image = np.zeros(orig_size, dtype=np.uint16)

    # Prepare all patch indices for iteration
    patch_indices = [(i, j) for i in range(num_patches_0) for j in range(num_patches_1)]

    # --- Batch process all patches for prediction ---
    for batch_start in range(0, total_patches, batch_size):
        batch_end = min(batch_start + batch_size, total_patches)
        batch = patch_indices[batch_start:batch_end]
        batch_imgs = []
        # Prepare batch of patch images for model
        for i, j in batch:
            patch_img = transform(patches_array[i, j, 0])
            batch_imgs.append(patch_img)
        batch_imgs = [img.to(device) for img in batch_imgs]

        # Run model prediction on batch of patches
        with torch.no_grad():
            pred_batch_info = model(batch_imgs)

        # --- Process each patch in the batch ---
        for idx_in_batch, (i, j) in enumerate(batch):
            divide_sum += 1
            # Get predictions for current patch
            pred_patch_info = pred_batch_info[idx_in_batch]
            pred_confidences = pred_patch_info["scores"].detach().cpu().numpy()
            # Indices of predictions above confidence threshold
            pred_t = [index for index, x in enumerate(pred_confidences) if x > confidence_thresh]
            # Get binary masks for predicted patches above threshold
            pred_masks = (pred_patch_info["masks"] > confidence_thresh).detach().cpu().numpy()
            if len(pred_t) > 0:
                pred_t_val = pred_t[-1]
                masks = pred_masks[: pred_t_val + 1]
            else:
                pred_t_val = 0
                masks = []
            # Initialize empty mask for this patch
            new_mask_expanded = np.expand_dims(np.zeros((patches_array.shape[3], patches_array.shape[4])), axis=-1)
            storage.arrays_to_unpatch[e][i, j, 0] = new_mask_expanded
            storage.confidence_scores[e] += 0
            # If there are valid masks, combine them and average confidence
            if len(masks) != 0:
                count = 0
                for indx in range(len(masks)):
                    new_mask_expanded = np.expand_dims(pred_masks[indx], axis=-1)
                    storage.confidence_scores[e] += pred_confidences[indx]
                    count += 1
                    # Combine masks using logical OR
                    storage.arrays_to_unpatch[e][i, j, 0] = np.ma.mask_or(
                        storage.arrays_to_unpatch[e][i, j, 0], np.array(new_mask_expanded).astype(np.uint8)
                    )
                storage.confidence_scores[e] = storage.confidence_scores[e] / count
                m = storage.arrays_to_unpatch[e][i, j, 0]
                m = (m * 255).astype(np.uint8)

            # Binarize the mask (ensure only 0 or 1 values)
            storage.arrays_to_unpatch[e][i, j, 0][storage.arrays_to_unpatch[e][i, j, 0] != 0] = 1
            # Place the patch mask into the correct location in the reconstructed image
            row_start = i * step
            row_end = row_start + patch_size
            col_start = j * step
            col_end = col_start + patch_size
            reconstructed_image[row_start:row_end, col_start:col_end] += storage.arrays_to_unpatch[e][i, j, 0].astype(
                np.uint16
            )

            e += 1  # Move to next patch

    # --- Finalize reconstruction ---
    # Keep raw reconstructed counts for inspection
    storage.raw_reconstructed = reconstructed_image.copy()
    # Binarize the reconstructed image (only 0 or 1 values)
    binarized = (reconstructed_image != 0).astype(np.uint8)
    # Convert to grayscale-like array (H,W)
    binarized_img = np.array(Image.fromarray(binarized).convert("L"))
    # Store the reconstructed mask in the output array (shape: (1,H,W))
    storage.output_array = (
        np.concatenate((storage.output_array, np.expand_dims(binarized_img, axis=0)), axis=0)
        if storage.output_array is not None
        else np.expand_dims(binarized_img, axis=0)
    )
    # Compute mean confidence score across all patches
    try:
        storage.output_confidence_score = (
            float(np.sum(storage.confidence_scores) / divide_sum) if divide_sum > 0 else 0.0
        )
    except Exception:
        storage.output_confidence_score = 0.0

    return storage


############################################
### SHARED FUNCTIONS FOR TESTING A MODEL ###


# Get image with black borders so its dimensions are divisible by patch_size and step (for full-image inference)
def return_image_black_borders(image, patch_size, step):
    # get number of patches needed along each dimension
    num_patches_0 = int(np.ceil((image.shape[0] - patch_size) / step + 1))
    num_patches_1 = int(np.ceil((image.shape[1] - patch_size) / step + 1))
    blackBorder_size_total_0 = abs(int((num_patches_0 - 1) * step + patch_size) - image.shape[0])
    blackBorder_size_total_1 = abs(int((num_patches_1 - 1) * step + patch_size) - image.shape[1])

    # For mask images: output shape (H, W, 1), for image: (H, W, 3)
    out_shape = (
        int(image.shape[0] + blackBorder_size_total_0),
        int(image.shape[1] + blackBorder_size_total_1),
        image.shape[2],
    )
    image_with_borders = np.zeros(out_shape, dtype=image.dtype)
    image_with_borders[: image.shape[0], : image.shape[1], :] = image
    # Ensure image is a numpy array with type uint8
    image_with_borders = np.array(image_with_borders).astype(np.uint8)
    # If mask is single channel, squeeze last dim and threshold to binary for consistency
    if image_with_borders.ndim == 3 and image_with_borders.shape[2] == 1:
        image_with_borders = image_with_borders.squeeze(-1)
        image_with_borders = (image_with_borders >= 125).astype(np.uint8) * 255

    return image_with_borders


# Crop predicted binary mask back to the original image size (for saving predictions)
def remove_black_borders(pred_obj, og_mask_image=None, og_rgb_image=None):
    # Original image must be provided to get target shape (H,W)
    if og_rgb_image is None:
        raise ValueError("original_image must be provided to remove black borders!")
    # Get original image dimensions (H,W)
    orig_h, orig_w = og_rgb_image.shape[0], og_rgb_image.shape[1]

    # Set ground-truth mask crop (if provided)
    og_mask_cropped = None
    if og_mask_image is not None:
        og_mask_cropped = og_mask_image[:orig_h, :orig_w]
        # remove last dim (single-channel squeezed dimension)
        if og_mask_cropped.ndim == 3 and og_mask_cropped.shape[2] == 1:
            og_mask_cropped = og_mask_cropped.squeeze(-1)

    pred_obj_cropped = None
    try:
        if isinstance(pred_obj, (list, tuple)):
            # Semantic: supports (binary_mask, raw_counts) or (binary_mask, raw_counts, gray_map)
            try:
                if len(pred_obj) == 3:
                    binary, raw, gray = pred_obj
                    # Crop predicted binary mask
                    binary_c = binary[:orig_h, :orig_w]
                    try:  # crop raw counts map
                        raw_c = raw[:orig_h, :orig_w]
                    except Exception:
                        raw_c = raw
                    try:  # crop gray map if available
                        gray_c = gray[:orig_h, :orig_w]
                    except Exception:
                        gray_c = gray
                    pred_obj_cropped = (binary_c, raw_c, gray_c)
            except Exception:
                # If cropping fails, only crop first element (predicted binary mask)
                try:
                    binary_c = pred_obj[0][:orig_h, :orig_w]
                    pred_obj_cropped = (binary_c,)
                except Exception:
                    pred_obj_cropped = pred_obj
        elif hasattr(pred_obj, "output_array"):
            # Storage-like object from Mask R-CNN (MaskConf_storage)
            try:
                if pred_obj.output_array is not None and pred_obj.output_array.shape[0] >= 1:
                    cropped = pred_obj.output_array[0][:orig_h, :orig_w]
                    pred_obj.output_array = np.expand_dims(cropped, axis=0)
            except Exception:
                # leave pred_obj unchanged on failure
                pass
            pred_obj_cropped = pred_obj
        else:
            # Numpy array mask (H,W) from semantic segmentation models
            try:
                pred_obj_cropped = pred_obj[:orig_h, :orig_w]
            except Exception:
                pred_obj_cropped = pred_obj
    except Exception:
        pred_obj_cropped = pred_obj

    # NOTE: pred_obj_cropped can be a tuple, storage object, or array depending on input type
    return pred_obj_cropped, og_rgb_image, og_mask_cropped


# Post-processing: Remove small connected components (blobs) from a predicted mask
def blob_removal(pred_array, min_area=1000, min_width=50, min_height=50):
    """
    Args:
        pred_array: array-like image (expected to be single-channel mask or 3-channel coloured mask)
        min_area: minimum area in pixels to keep a component
        min_width: minimum width in pixels to keep a component
        min_height: minimum height in pixels to keep a component
    Returns:
        filteredImage: uint8 binary image with remaining components set to 255, background 0
    """
    # Convert to grayscale if necessary
    if pred_array is None:
        return None
    try:
        inputImage_mask = cv2.cvtColor(pred_array, cv2.COLOR_BGR2GRAY)
    except Exception:
        # If conversion fails (e.g. already single channel) try direct cast
        inputImage_mask = np.array(pred_array)
        if inputImage_mask.ndim == 3:
            inputImage_mask = inputImage_mask[..., 0]
    mask_image = inputImage_mask.copy()

    # componentStats[i] --> [0]=x ; [1]=y ; [2]=w ; [3]=h ; [4]=area
    componentsNumber, labeledImage, componentStats, _ = cv2.connectedComponentsWithStats(mask_image, 4, cv2.CV_32S)
    remainingComponentLabels = []
    for i in range(1, componentsNumber):
        area = componentStats[i][4]
        width = componentStats[i][2]
        height = componentStats[i][3]
        # Keep if area is large enough, and either width or height is large enough
        if area >= min_area and (width > min_width or height > min_height):
            remainingComponentLabels.append(i)
    # Create filtered image with only remaining components binarized to 255 (background 0)
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels), 255, 0).astype("uint8")
    return filteredImage


# Color the predicted binary mask for visualization
def get_coloured_mask(predicted_mask, color_rgb=[255, 255, 255]):
    """
    Create an RGB image (H,W,3) colouring pixels where predicted_mask == 1
    The input mask can be a binary 2D array or boolean array
    """
    r = np.zeros_like(predicted_mask).astype(np.uint8)
    g = np.zeros_like(predicted_mask).astype(np.uint8)
    b = np.zeros_like(predicted_mask).astype(np.uint8)
    mask_cond = predicted_mask == 1
    r[mask_cond], g[mask_cond], b[mask_cond] = color_rgb
    # Return a stacked RGB image of the predicted mask
    return np.stack([r, g, b], axis=2)


# Get a two-panel plot of original image and prediction, with evaluation metrics (if ground-truth mask provided)
def eval_and_plot(
    name_without_ext,
    og_rgb_image,
    og_mask,
    binary_prediction_orig,
    binary_prediction_blobRmv,
    prediction_dir,
    eval_info=None,
):
    """
    Compute optional evaluation metrics (Dice, IoU) if ground-truth mask is provided,
    find the variant with better metrics, write an evaluation markdown file, and save a two-panel plot.
    For images where og_mask is None, the left panel will show the original RGB image and the right panel the predicted mask.
    Returns:
        dict: summary with keys 'dice' and 'iou' (None when not computed) and the better variant name
    """
    eval_lines = []
    if eval_info is not None:
        # Add any already existing eval info if provided
        for k, v in eval_info.items():
            eval_lines.append(f"{k}: {v}")

    # Default choices for plotting/evaluation (usually blob-removed variant preferred)
    better_variant = "Blob-removed Prediction" if binary_prediction_blobRmv is not None else "Original Prediction"
    chosen_image_for_plot = (
        binary_prediction_blobRmv if binary_prediction_blobRmv is not None else binary_prediction_orig
    )

    # Initialize chosen metrics (None if GT is not provided)
    iou = None
    dice = None
    chosen_iou = None
    chosen_dice = None

    if og_mask is not None:
        # Ensure mask is binary 0/1
        try:
            mask_norm = og_mask.copy()
            mask_norm = mask_norm / 255.0 if mask_norm.max() > 1 else mask_norm
            mask_norm[mask_norm > 0] = 1
        except Exception:
            mask_norm = og_mask

        # Convert predictions to binary 0/1 for metric computation
        pred_bin_blob = (binary_prediction_blobRmv > 0).astype(np.uint8)
        pred_bin_orig = (binary_prediction_orig > 0).astype(np.uint8)

        # Compute metrics for both variants when available
        if pred_bin_blob is not None:
            iou_blob = compute_iou(mask_norm, pred_bin_blob)
            dice_blob = compute_dice(mask_norm, pred_bin_blob)
            eval_lines.append(f"Blob-removed IoU: {iou_blob:.4f}")
            eval_lines.append(f"Blob-removed Dice coefficient: {dice_blob:.4f}")

        if pred_bin_orig is not None:
            iou_orig = compute_iou(mask_norm, pred_bin_orig)
            dice_orig = compute_dice(mask_norm, pred_bin_orig)
            eval_lines.append(f"Original-pred IoU: {iou_orig:.4f}")
            eval_lines.append(f"Original-pred Dice coefficient: {dice_orig:.4f}")

        # Default to blob-removed variant
        chosen_iou = iou_blob
        chosen_dice = dice_blob
        # If the original image's IoU is greater, pick original
        if iou_blob is None or iou_orig > iou_blob:
            better_variant = "Original Prediction"
            chosen_image_for_plot = binary_prediction_orig
            chosen_iou = iou_orig
            chosen_dice = dice_orig
        eval_lines.append(f"Better variant (by IoU): {better_variant}")
        # Expose iou/dice values to outer scope
        dice = chosen_dice
        iou = chosen_iou

    elif eval_info is not None:
        eval_lines.append("No ground truth mask available (ogmask not provided)")

    # Write full evaluation into markdown file if eval_info is provided
    if eval_info is not None:
        with open(os.path.join(prediction_dir, f"evaluation_info_{name_without_ext}.md"), "w") as f:
            for line in eval_lines:
                f.write(line + "\n")

    # --- Create two-panel plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    if og_mask is not None:  # ground truth mask provided
        # Left side: ground truth mask
        axes[0].imshow(og_mask, cmap="gray")
        axes[0].set_title(f"{name_without_ext} Ground Truth Mask")
        axes[0].axis("off")
        # Right side: show the chosen predicted mask (the one with better IoU)
        axes[1].imshow(chosen_image_for_plot, cmap="gray")
        axes[1].axis("off")
        right_title = (
            f"{better_variant} (IoU: {iou:.4f}, Dice: {dice:.4f})"
            if (iou is not None and dice is not None)
            else f"{better_variant} (metrics unavailable)"
        )
        axes[1].set_title(right_title)
    else:  # no ground truth mask provided for inference image
        # Left: original RGB
        axes[0].imshow(og_rgb_image)
        axes[0].set_title(f"{name_without_ext} Original Image")
        axes[0].axis("off")
        # Right: predicted mask (default to blob-removed)
        axes[1].imshow(chosen_image_for_plot, cmap="gray")
        axes[1].set_title(f"{better_variant} (no GT)")
        axes[1].axis("off")

    fig.tight_layout()
    plt.savefig(os.path.join(prediction_dir, f"{name_without_ext}_mask_plot.png"))
    plt.close(fig)

    return {"dice": dice, "iou": iou}, better_variant
