import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


class PendulumFrameDataset(Dataset):
    """
    Returns individual frames (not sequences) for VAE training.
    Each item: tensor (3, H, W) float32 in [0, 1].
    """

    def __init__(self, dataset_dir: str = "dataset/double_pendulum"):
        self.files = sorted(Path(dataset_dir).glob("traj_*.npz"))
        assert self.files, f"No .npz files found in {dataset_dir}"

        # Pre-index: (file_idx, frame_idx) for every frame
        first = np.load(self.files[0])
        self.n_frames = first["frames"].shape[0]
        self.index = [
            (fi, t)
            for fi in range(len(self.files))
            for t in range(self.n_frames)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, t = self.index[idx]
        data = np.load(self.files[fi])
        frame = data["frames"][t]                        # (H, W, 3) uint8
        x = torch.from_numpy(frame).permute(2, 0, 1)    # (3, H, W)
        return x.float() / 255.0


class PendulumSeqDataset(Dataset):
    """
    Returns full sequences for dynamics model training.
    Each item: tensor (T, 3, H, W) float32 in [0, 1], states (T, 4).
    """

    def __init__(self, dataset_dir: str = "dataset/double_pendulum"):
        self.files = sorted(Path(dataset_dir).glob("traj_*.npz"))
        assert self.files, f"No .npz files found in {dataset_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        frames = torch.from_numpy(data["frames"]).permute(0, 3, 1, 2).float() / 255.0
        states = torch.from_numpy(data["states"]).float()
        return frames, states


def make_seq_dataloaders(
    dataset_dir: str = "dataset/double_pendulum",
    batch_size: int = 16,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
):
    """Dataloaders de séquences (frames, states) pour l'entraînement JEPA."""
    dataset = PendulumSeqDataset(dataset_dir)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def make_dataloaders(
    dataset_dir: str = "dataset/double_pendulum",
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
):
    dataset = PendulumFrameDataset(dataset_dir)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
