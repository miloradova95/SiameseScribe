import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tse_medieval.data_io.classes.pen_flourishing_dataset import PenFlourishingDataset


def patch_dataset(config: dict) -> Dataset:
    """
    Create dataset of patches from provided path.

    :param config: The configuration dictionary containing function parameters.
    :type config: dict
    :return: The patch dataset.
    :rtype: torch.utils.data.Dataset
    """
    # Configure data transforms
    transformation_function = v2.Compose(
        [
            v2.Resize(size=config.get("patch_size")),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        ]
    )

    # Create pytorch dataset
    return PenFlourishingDataset(
        **config, transformation_function=transformation_function
    )
