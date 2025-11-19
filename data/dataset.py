"""Defines the parking slot dataset for directional marking point detection."""
import json
import os
import os.path
import cv2 as cv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from data.struct import MarkingPoint


# class ParkingSlotDataset(Dataset):
#     """Parking slot dataset."""
#     def __init__(self, root):
#         super(ParkingSlotDataset, self).__init__()
#         self.root = root
#         self.sample_names = []
#         self.image_transform = ToTensor()
#         for file in os.listdir(root):
#             if file.endswith(".json"):
#                 self.sample_names.append(os.path.splitext(file)[0])

#     def __getitem__(self, index):
#         name = self.sample_names[index]
#         image = cv.imread(os.path.join(self.root, name+'.jpg'))

#         image = self.image_transform(image)
#         marking_points = []
#         with open(os.path.join(self.root, name + '.json'), 'r') as file:
#             for label in json.load(file):
#                 marking_points.append(MarkingPoint(*label))
#         return image, marking_points

#     def __len__(self):
#         return len(self.sample_names)


import os
import json
import random
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ParkingSlotDataset(Dataset):
    """Parking slot dataset with temporal-aware augmentation."""
    def __init__(self, root, augment=True):
        self.root = root
        self.augment = augment
        self.sample_names = []
        self.to_tensor = ToTensor()

        for file in os.listdir(root):
            if file.endswith(".json"):
                self.sample_names.append(os.path.splitext(file)[0])

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        name = self.sample_names[idx]
        img = cv.imread(os.path.join(self.root, name + ".jpg"))

        # ==== split left/right frame ====
        pre = img[:, :512, :]
        now = img[:, 512:, :]

        # ==== load labels ====
        marking_points = []
        with open(os.path.join(self.root, name + ".json")) as f:
            for label in json.load(f):
                marking_points.append(MarkingPoint(*label))

        if self.augment:
            pre, now, marking_points = self.apply_augmentation_safe(pre, now, marking_points)

        # 拼接回去
        img = np.concatenate([pre, now], axis=1)

        img = self.to_tensor(img)
        return img, marking_points

    # ================================================================
    #                  🔥 Temporal-aware augmentation
    # ================================================================

    def apply_augmentation_safe(self, pre, now, points):
        """
        Safe augmentations that DO NOT change spatial coordinates of points.
        - pre, now: HxWxC uint8 images (OpenCV BGR)
        - points: list of MarkingPoint (unused here; needed only if you want cover-avoid)
        Returns augmented (pre, now, points) without modifying points in-place.
        """

        H, W, _ = now.shape

        # ---- 1) synchronized photometric (both frames) ----
        if random.random() < 0.5:
            # brightness / contrast
            alpha = 1.0 + (random.random() * 0.2 - 0.1)  # ±10%
            beta = random.randint(-15, 15)
            pre = cv.convertScaleAbs(pre, alpha=alpha, beta=beta)
            now = cv.convertScaleAbs(now, alpha=alpha, beta=beta)

        if random.random() < 0.3:
            # gaussian blur with small kernel
            k = random.choice([3,5])
            pre = cv.GaussianBlur(pre, (k, k), 0)
            now = cv.GaussianBlur(now, (k, k), 0)

        # ---- 2) per-now photometric / noise variations (simulate frame diff) ----
        if random.random() < 0.7:
            alpha = 1.0 + (random.random() * 0.12 - 0.06)
            beta = random.randint(-8, 8)
            now = cv.convertScaleAbs(now, alpha=alpha, beta=beta)

        if random.random() < 0.5:
            # additive gaussian noise
            sigma = random.uniform(1.5, 6.0)
            noise = np.random.randn(*now.shape) * sigma
            now = np.clip(now + noise, 0, 255).astype(np.uint8)

        if random.random() < 0.4:
            # motion blur (directional)
            k = random.choice([7,9,11])
            # create linear kernel
            kernel = np.zeros((k, k))
            if random.random() < 0.5:
                # horizontal-ish
                kernel[k//2, :] = np.ones(k)
            else:
                kernel[:, k//2] = np.ones(k)
            kernel = kernel / kernel.sum()
            now = cv.filter2D(now, -1, kernel)

        if random.random() < 0.3:
            # jpeg compression artifacts
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), random.randint(50, 95)]
            _, encimg = cv.imencode('.jpg', now, encode_param)
            now = cv.imdecode(encimg, 1)

        # ---- 3) random local occlusion (cutout) on 'now' ----
        # We'll try to avoid covering keypoints if points are given and normalized in frame coords.
        if random.random() < 0.3:
            num_squares = random.randint(1, 3)
            for _ in range(num_squares):
                w = random.randint(10, 40)
                h = random.randint(10, 40)
                x = random.randint(0, W - w)
                y = random.randint(0, H - h)
                # optional: check keypoints to avoid major occlusion (skipped here)
                now[y:y+h, x:x+w, :] = np.random.randint(0, 255, (1,1,3), dtype=np.uint8)

        # ---- 4) slight color jitter in HSV (only now) ----
        if random.random() < 0.25:
            hsv = cv.cvtColor(now, cv.COLOR_BGR2HSV).astype(np.int32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.9, 1.15), 0, 255)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-5, 5)) % 180
            now = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

        return pre, now, points