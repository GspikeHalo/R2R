import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class NpyDomainFolder(Dataset):
    """
    Dataset for loading .npy image tensors arranged in domain‐split folders.

    Expects directory structure under `root` of the form:
        root/
          trainA/      <-- .npy files for domain A training
          trainB/      <-- .npy files for domain B training
          testA/       <-- .npy files for domain A testing
          testB/       <-- .npy files for domain B testing

    Parameters
    ----------
    root : str
        Base path to the dataset.
    domain : str
        Domain letter, e.g. 'A' or 'B' (case‐insensitive).
    split : str
        One of 'train', 'test', or 'val'.
    transform : callable, optional
        A function/transform that takes in a PIL Image or Tensor
        and returns a transformed version.
    """

    def __init__(self, root, domain='A', split='train', transform=None):
        super().__init__()
        subdir = os.path.join(split, domain)
        self.folder = os.path.join(root, subdir)
        if not os.path.isdir(self.folder):
            raise ValueError(f"Directory not found: {self.folder}")

        # Collect all .npy files in the subdirectory
        pattern = os.path.join(self.folder, '*.npy')
        self.files = sorted(glob.glob(pattern))
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {self.folder}")

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr  = np.load(path)            # load ndarray
        tensor = torch.from_numpy(arr).float()  # convert to Tensor
        # Optionally, you could convert to PIL Image if your transforms expect that:
        # from PIL import Image
        # tensor = torch.from_numpy(arr)
        # img = Image.fromarray(arr)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor