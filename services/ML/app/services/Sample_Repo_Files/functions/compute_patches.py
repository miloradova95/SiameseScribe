from typing import Literal, Union

import numpy as np
from patchify import patchify
import PIL


def compute_patches(
    image: Union[PIL.Image.Image, np.ndarray],
    color_mode: Literal["RGB", "L"],
    patch_size: tuple[int, int],
    step_size: int,
) -> tuple[np.ndarray, list]:
    """
    Generate patches from an image.

    :param image: Input image (PIL.Image.Image or ndarray).
    :type image: Union[PIL.Image.Image, np.ndarray]
    :param color_mode: Image color mode, RGB for color or other for grayscale.
    :type color_mode: Literal["RGB", "L"]
    :param patch_size: Size of each patch as (height, width).
    :type patch_size: tuple[int, int]
    :param step_size: Step size for sliding window.
    :type step_size: int
    :return: Array of image patches and their positions.
    :rtype: tuple[np.ndarray, list]
    """
    image_array = np.array(image)
    if color_mode == "RGB":
        patch_dims = (patch_size[0], patch_size[1], 3)
    else:
        patch_dims = (patch_size[0], patch_size[1])

    # Apply patchify for patch computation
    patches_grid = patchify(image_array, patch_dims, step=step_size)
    n_rows, n_cols = patches_grid.shape[0], patches_grid.shape[1]
    flat_patches = patches_grid.reshape(-1, *patch_dims)

    # Calculate patch_positions based on the defined grid parameters
    positions = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = j * step_size
            y = i * step_size
            positions.append((x, y))

    return flat_patches, positions
