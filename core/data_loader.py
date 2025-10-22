"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
import os
import random

from munch import Munch
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms


def listdir(dname):
    fnames = list(Path(dname).rglob('*.npy'))
    fnames.sort()
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        arr = np.load(str(fname))
        img = torch.from_numpy(arr).float()
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

class NpyFolder(data.Dataset):
    def __init__(self, root, transform=None, fixed_filenames=None):
        root = Path(root)
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        samples = []
        for cls in classes:
            cls_dir = root / cls
            for fn in cls_dir.rglob('*.npy'):
                name = fn.name
                if fixed_filenames is not None:
                    allow_list = fixed_filenames.get(cls, [])
                    if (name not in allow_list) and (Path(name).stem not in allow_list):
                        continue
                samples.append((fn, self.class_to_idx[cls]))
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        arr = np.load(str(path))               # shape (4, H, W) or similar
        img = torch.from_numpy(arr).float()    # convert to float tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples, self.targets = self._make_dataset(root)

    def _make_dataset(self, root):
        domains = sorted(os.listdir(root))
        fnames1, fnames2, labels = [], [], []
        for idx, domain in enumerate(domains):
            class_dir = os.path.join(root, domain)
            if not os.path.isdir(class_dir):
                continue
            files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            paths = [os.path.join(class_dir, f) for f in files]
            if len(paths) == 0:
                continue
            fnames1 += paths
            fnames2 += random.sample(paths, len(paths))
            labels += [idx] * len(paths)
        return list(zip(fnames1, fnames2)), labels

    def __getitem__(self, index):
        path1, path2 = self.samples[index]
        label = self.targets[index]
        arr1 = np.load(path1)
        arr2 = np.load(path2)
        img1 = torch.from_numpy(arr1).float()
        img2 = torch.from_numpy(arr2).float()
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label

    def __len__(self):
        return len(self.targets)

class PairedNpyDataset(data.Dataset):
    def __init__(self, root: str, domains: list[str], transform=None):
        self.root = Path(root)
        self.domains = domains
        self.transform = transform
        self.dirs = [self.root / d for d in domains]

        sets = [set(p.name for p in d.glob("*.npy")) for d in self.dirs]
        common = sorted(set.intersection(*sets))

        if not common:
            raise RuntimeError(f"No common .npy among {domains}")

        self.pairs = [
            [d / fname for d in self.dirs]
            for fname in common
        ]
        self.filenames=common

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        paths = self.pairs[idx]
        imgs = {}
        for domain, p in zip(self.domains, paths):
            arr = np.load(str(p)) # (4, H, W)
            img = torch.from_numpy(arr).float()
            if self.transform:
                img = self.transform(img)
            imgs[domain] = img
        return imgs, self.filenames[idx]


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4, fixed_filenames=None):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    if (which == 'source') and (fixed_filenames is not None):
        transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],
                                 std =[0.5, 0.5, 0.5, 0.5]),
        ])
    else:
        crop = transforms.RandomResizedCrop(
            img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
        rand_crop = transforms.Lambda(
            lambda x: crop(x) if random.random() < prob else x)

        transform = transforms.Compose([
            rand_crop,
            transforms.Resize([img_size, img_size]),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5]),
        ])

    if which == 'source':
        dataset = NpyFolder(root, transform, fixed_filenames)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    if (which == 'source') and (fixed_filenames is not None):
        return data.DataLoader(dataset=dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers,
                               pin_memory=True,
                               drop_last=True)
    else:
        sampler = _make_balanced_sampler(dataset.targets)
        return data.DataLoader(dataset=dataset,
                               batch_size=batch_size,
                               sampler=sampler,
                               num_workers=num_workers,
                               pin_memory=True,
                               drop_last=True)

def get_test_loader(root, domains, img_size=256, batch_size=32,
                    shuffle=False, num_workers=4):
    print('Preparing DataLoader for the generation phase (4-ch npy)...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],
                             std =[0.5, 0.5, 0.5, 0.5]),
    ])

    paired_ds = PairedNpyDataset(
        root=root,
        domains=domains,
        transform=transform
    )

    return data.DataLoader(dataset=paired_ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)

class InputFetcher:
    def __init__(self, loader, loader_ref=None, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})

def build_train_ref_pool(root: str):
    """
    构建参考池（reference pool）：
    返回 domains(list[str]) 与 pool(dict[str, list[str]])。
    每个域对应该域下所有 .npy 文件的绝对路径。
    """
    root_p = Path(root)
    domains = sorted([p.name for p in root_p.iterdir() if p.is_dir()])
    pool = {}
    for d in domains:
        ddir = root_p / d
        paths = sorted(str(p.resolve()) for p in ddir.rglob('*.npy'))
        pool[d] = paths
    return domains, pool


def sample_k_from_pool(pool: dict, domains: list, y: torch.Tensor, K: int,
                       transform, device: torch.device):
    """
    从池中按域抽取 K 张参考（per-sample），并应用给定 transform。
    输入：
      - pool: {domain -> [paths]}
      - domains: 与 y 的索引一致的域名列表
      - y: [B]，域索引（long）
      - K: 每个样本抽 K 张
      - transform: 与训练/评测一致的 transform（Compose）
      - device: 目标设备
    输出：
      - x_ref: [B, K, 4, H, W]（float, 已 transform, 在 device）
    """
    B = y.shape[0]
    out = []
    for b in range(B):
        dname = domains[int(y[b].item())]
        paths = pool.get(dname, [])
        if not paths:
            raise RuntimeError(f"No ref paths for domain={dname}")
        # 随机选择 K 条路径（可重复抽样）
        idxs = torch.randint(low=0, high=len(paths), size=(K,))
        imgs = []
        for j in idxs.tolist():
            arr = np.load(paths[j])
            img = torch.from_numpy(arr).float()
            if transform is not None:
                img = transform(img)
            imgs.append(img)
        xk = torch.stack(imgs, dim=0)  # [K,4,H,W]
        out.append(xk)
    x_ref = torch.stack(out, dim=0).to(device)  # [B,K,4,H,W]
    return x_ref
