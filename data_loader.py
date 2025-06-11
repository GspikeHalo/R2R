from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np
import torch.nn.functional as F

class MyData(data.Dataset):
    """Dataset class for the custom 4-channel .npy dataset."""
    def __init__(self, image_dir, attr_path, selected_attrs, mode, crop_size=178, image_size=128):
        """Initialize and preprocess the custom dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.mode = mode
        self.crop_size = crop_size
        self.image_size = image_size
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the attribute file (CelebA-style) for the custom dataset."""

        if self.mode == 'train':
            lines = [line.rstrip() for line in open(self.attr_path, 'r')]
            all_attr_names = lines[1].split()
            for i, attr_name in enumerate(all_attr_names):
                self.attr2idx[attr_name] = i
                self.idx2attr[i] = attr_name
            # attr2idx: {cameraA: 0, cameraB: 1}
            # idx2attr: {0: cameraA, 1: cameraA}
            lines = lines[2:]
            # Use all data for training (no hold-out for test)
            random.seed(1234)
            random.shuffle(lines)
            for line in lines:
                split = line.split()
                filename = split[0]
                values = split[1:]
                label = [] # 保存的是一行中的attr的true or false
                for attr_name in self.selected_attrs:
                    idx = self.attr2idx[attr_name]
                    label.append(values[idx] == '1')  # True for '1', False for '-1'
                self.train_dataset.append([filename, label])
        else:
            d0 = os.path.join(self.image_dir, self.selected_attrs[0])
            d1 = os.path.join(self.image_dir, self.selected_attrs[1])
            files0 = sorted([f for f in os.listdir(d0) if f.endswith('.npy')])
            files1 = set([f for f in os.listdir(d1) if f.endswith('.npy')])
            for fn in files0:
                if fn in files1:
                    p0 = os.path.join(d0, fn)
                    p1 = os.path.join(d1, fn)
                    # one-hot 标签
                    c0 = torch.zeros(len(self.selected_attrs), dtype=torch.float32)
                    c1 = torch.zeros(len(self.selected_attrs), dtype=torch.float32)
                    c0[0] = 1.0
                    c1[1] = 1.0
                    self.test_dataset.append((p0, p1, c0, c1))

    def __getitem__(self, index):
        """Return a single image (train) or a pair of images (test) with their labels."""
        if self.mode == 'train':
            # 添加水平翻转试试
            filename, label = self.train_dataset[index]
            img = np.load(os.path.join(self.image_dir, filename)).astype(np.float32)
            if random.random() < 0.5:
                img = np.flip(img, axis=2).copy()

            C, H, W = img.shape
            # Center crop if larger than crop_size
            if H >= self.crop_size and W >= self.crop_size:
                cy = (H - self.crop_size) // 2
                cx = (W - self.crop_size) // 2
                img = img[:, cy:cy + self.crop_size, cx:cx + self.crop_size]
                H, W = self.crop_size, self.crop_size
            # Resize to target image_size if needed
            if H != self.image_size or W != self.image_size:
                img_tensor = torch.from_numpy(img).unsqueeze(0)
                img_tensor = F.interpolate(img_tensor, size=(self.image_size, self.image_size), mode='bilinear',
                                           align_corners=False)
                img_tensor = img_tensor[0]
            else:
                img_tensor = torch.from_numpy(img)
            # Scale pixel values to [-1, 1]
            img_tensor = img_tensor * 2.0 - 1.0
            return img_tensor, torch.FloatTensor(label)
        else:
            p0, p1, c0, c1 = self.test_dataset[index]
            arr0 = np.load(p0).astype(np.float32)
            arr1 = np.load(p1).astype(np.float32)
            img0 = self._process(arr0)
            img1 = self._process(arr1)
            return img0, img1, c0, c1, p0, p1
    
    def _process(self, img):
        # img: [C,H,W]
        C, H, W = img.shape
        # 中心裁剪
        if H >= self.crop_size and W >= self.crop_size:
            cy, cx = (H - self.crop_size) // 2, (W - self.crop_size) // 2
            img = img[:, cy:cy+self.crop_size, cx:cx+self.crop_size]
        # 调整大小
        if img.shape[1] != self.image_size or img.shape[2] != self.image_size:
            t = torch.from_numpy(img).unsqueeze(0)
            t = F.interpolate(t, size=(self.image_size, self.image_size),
                              mode='bilinear', align_corners=False)[0]
            img = t.numpy()
        # 归一化到 [-1,1]
        img_t = torch.from_numpy(img) * 2.0 - 1.0
        return img_t


    def __len__(self):
        return self.num_images

def get_loader(image_dir, attr_path, selected_attrs, crop_size=256, image_size=256,
               batch_size=16, mode='train', num_workers=1):
    dataset = MyData(image_dir, attr_path, selected_attrs, mode, crop_size=crop_size, image_size=image_size)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=(mode == 'train'),
                           num_workers=num_workers)