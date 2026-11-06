"""
PyTorch Dataset and Augmentation Pipeline for 6-Channel Sentinel-2 Imagery.
"""

from typing import List, Dict, Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


def get_transforms(split: str = "train") -> A.Compose:
    """Returns spatial data augmentations.

    For satellite imagery, rotational invariance holds:
    Horizontal flips, vertical flips, and 90-degree rotations are physically valid.
    """
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    return A.Compose([])


class SatellitePatchDataset(Dataset):
    """PyTorch Dataset loading multispectral 6-channel Sentinel-2 patches

    and binary forest masks.

    Bands:
        0: B2 (Blue)
        1: B3 (Green)
        2: B4 (Red)
        3: B8 (NIR)
        4: B11 (SWIR 1)
        5: B12 (SWIR 2)
    """

    def __init__(
        self,
        tile_metadata: List[Dict],
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
    ):
        self.metadata = tile_metadata
        self.transform = transform

        # Normalization vectors (6 channels)
        if mean is not None and std is not None:
            self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
        else:
            self.mean = None
            self.std = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[idx]
        data = np.load(meta["file_path"])

        image = data["image"]  # Shape: (6, H, W)
        mask = data["mask"]    # Shape: (H, W)

        # Apply band-wise z-score normalization
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std

        # Albumentations expects (H, W, C) for image
        if self.transform is not None:
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc, mask=mask)
            image = np.transpose(augmented["image"], (2, 0, 1))
            mask = augmented["mask"]

        # Convert to PyTorch Tensors
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "tile_id": meta["tile_id"],
        }
