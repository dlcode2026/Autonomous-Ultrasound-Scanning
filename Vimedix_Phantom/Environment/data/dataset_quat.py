import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import os
import math
import random
import pandas as pd


def min_max_normalize(value, min_val, max_val, eps=1e-8):
    return (value - min_val) / (max_val - min_val + eps)

NUM_VARIANTS = 2   # 0 = original, 1 = mild Gaussian blur

def _apply_variant(img: Image.Image, variant: int) -> Image.Image:
    if variant == 0:
        return img
    elif variant == 1:
        radius = random.uniform(0.5, 1.5)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


class FlatUltrasoundDataset(Dataset):   

    NORM_MIN = np.array([-0.205, -0.487173,  0.198017,
                     -0.706294, -0.705661, -0.340682, -0.440026], dtype=np.float32)
    NORM_MAX = np.array([ 0.010570, -0.272387,  0.274926,
                      0.996069,  0.999752,  0.388672,  0.377515],  
                    dtype=np.float32)

    def __init__(self, image_dir: str, csv_path: str,
                 transform=None, augment: bool = False,
                 drop_missing: bool = True):

        self.image_dir = image_dir
        self.transform = transform
        self.augment   = augment

        # Load CSV
        df = pd.read_csv(csv_path).reset_index(drop=True)
        print(f"  Loaded CSV: {csv_path}  ({len(df)} rows)")

        if drop_missing:
            exists = df["filename"].apply(
                lambda f: os.path.exists(os.path.join(image_dir, f))
            )
            n_missing = (~exists).sum()
            if n_missing > 0:
                print(f"  Dropping {n_missing} rows with missing images")
            df = df[exists].reset_index(drop=True)

        self.df        = df
        self._base_len = len(self.df)

        aug_str = (f" → ×{NUM_VARIANTS} augmented = {self._base_len * NUM_VARIANTS}"
                   if augment else "")
        print(f"  FlatUltrasoundDataset: {self._base_len} poses{aug_str}")
        print(f"  NORM_MIN (pos): {self.NORM_MIN[:3]}")
        print(f"  NORM_MAX (pos): {self.NORM_MAX[:3]}")

    def __len__(self):
        return self._base_len * NUM_VARIANTS if self.augment else self._base_len

    def _get_pose(self, row) -> np.ndarray:
        """Extract [x, y, z, qx, qy, qz, qw] from a CSV row."""
        return np.array([
            float(row["pos_x"]),    float(row["pos_y"]),    float(row["pos_z"]),
            float(row["qx"]), float(row["qy"]),
            float(row["qz"]), float(row["qw"]),
        ], dtype=np.float32)

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        out = np.zeros(7, dtype=np.float32)
        for i in range(3):
            out[i] = (2 * (raw[i] - self.NORM_MIN[i])
                      / (self.NORM_MAX[i] - self.NORM_MIN[i] + 1e-8) - 1)
        out[3:] = raw[3:]
        return out

    def __getitem__(self, idx):
        if self.augment:
            base_idx = idx % self._base_len
            variant  = idx // self._base_len
        else:
            base_idx = idx
            variant  = 0

        row      = self.df.iloc[base_idx]
        img_path = os.path.join(self.image_dir, row["filename"])

        img = Image.open(img_path).convert("L")
        img = _apply_variant(img, variant)

        raw_pose  = self._get_pose(row)
        norm_pose = self._normalize(raw_pose)
        label     = torch.tensor(norm_pose, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_raw_pose(self, idx: int) -> np.ndarray:
        base_idx = idx % self._base_len
        return self._get_pose(self.df.iloc[base_idx])

    @property
    def normalization_params(self):
        return {"min": self.NORM_MIN, "max": self.NORM_MAX}


def get_all_dataset_paths(base_dir, sets):
    dataset_pairs = []
    for s in sets:
        csv_path = os.path.join(base_dir, s, "positions.csv")
        img_dir  = os.path.join(base_dir, s, "images")
        if os.path.exists(csv_path) and os.path.exists(img_dir):
            dataset_pairs.append((img_dir, csv_path))
            print(f"Found valid dataset pair: {s}")
        else:
            print(f"Warning: missing csv or images folder for {s}")
    return dataset_pairs


class UltrasoundDataset(Dataset):
    def __init__(self, image_folder, csv_path, transform=None,
                 normalization_params=None, normalization_method="domain_aware",
                 calculate_params=False, augment=False):

        self.image_folder         = image_folder
        self.transform            = transform
        self.normalization_method = normalization_method
        self.augment              = augment

        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_filenames = self.df["filename"].tolist()
        self._base_len = len(self.image_filenames)

        self.normalization_params = normalization_params

    def _get_pose(self, row):
        return np.array([
            float(row["pos_x"]),    float(row["pos_y"]),    float(row["pos_z"]),
            float(row["orient_x"]), float(row["orient_y"]),
            float(row["orient_z"]), float(row["orient_w"]),
        ], dtype=np.float32)

    def normalize_values(self, values):
        if self.normalization_params is None:
            return values
        mn  = self.normalization_params["min"]
        mx  = self.normalization_params["max"]
        out = np.zeros(7, dtype=np.float32)
        for i in range(3):
            out[i] = 2 * (values[i] - mn[i]) / (mx[i] - mn[i] + 1e-8) - 1
        out[3:] = values[3:]
        return out

    def __len__(self):
        return self._base_len * NUM_VARIANTS if self.augment else self._base_len

    def __getitem__(self, idx):
        base_idx = idx % self._base_len
        variant  = (idx // self._base_len) if self.augment else 0

        row      = self.df.iloc[base_idx]
        img_path = os.path.join(self.image_folder, row["filename"])
        img      = Image.open(img_path).convert("L")
        img      = _apply_variant(img, variant)

        raw   = self._get_pose(row)
        norm  = self.normalize_values(raw)
        label = torch.tensor(norm, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, label


class MultiUltrasoundDataset(Dataset):

    '''NORM_MIN = np.array([-0.205, -0.450, 0.210,
                         -0.706294, -0.705661, -0.340682, -0.440026],
                        dtype=np.float32)
    NORM_MAX = np.array([-0.010, -0.344, 0.225,
                           0.996069,  0.999752,  0.388672,  0.377515],
                        dtype=np.float32)'''

    NORM_MIN = np.array([-0.205, -0.487173,  0.198017,
                     -0.706294, -0.705661, -0.340682, -0.440026], dtype=np.float32)
    NORM_MAX = np.array([ 0.010570, -0.272387,  0.274926,
                      0.996069,  0.999752,  0.388672,  0.377515],
                    dtype=np.float32)

    def __init__(self, base_image_dir, sets, transform=None,
                 normalization_method="domain_aware",
                 calculate_params=True, augment=False):

        self.augment = augment
        dataset_pairs = get_all_dataset_paths(base_image_dir, sets)
        if not dataset_pairs:
            raise ValueError("No valid dataset pairs found")

        norm_params = {"min": self.NORM_MIN, "max": self.NORM_MAX}

        self.datasets = []
        for img_dir, csv_path in dataset_pairs:
            ds = UltrasoundDataset(
                image_folder=img_dir,
                csv_path=csv_path,
                transform=transform,
                normalization_params=norm_params,
                normalization_method=normalization_method,
                augment=augment,
            )
            self.datasets.append(ds)

        self.dataset_offsets = [0]
        for ds in self.datasets:
            self.dataset_offsets.append(self.dataset_offsets[-1] + len(ds))

        print(f"MultiUltrasoundDataset: {self.dataset_offsets[-1]} total samples")

    def __len__(self):
        return self.dataset_offsets[-1]

    def __getitem__(self, idx):
        ds_idx = next(
            i for i, offset in enumerate(self.dataset_offsets[1:], 1)
            if offset > idx
        ) - 1
        return self.datasets[ds_idx][idx - self.dataset_offsets[ds_idx]]

    def get_normalization_params(self):
        return {"min": self.NORM_MIN, "max": self.NORM_MAX}
