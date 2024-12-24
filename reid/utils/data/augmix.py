# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reference implementation of AugMix's data augmentation method in numpy."""
from reid.utils.data.augmentations import augmentations
import numpy as np
from PIL import Image
import random
import math
import torch
# CIFAR-10 constants
# MEAN = [0.4914, 0.4822, 0.4465]
# STD = [0.2023, 0.1994, 0.2010]
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    image = image.transpose(2, 0, 1)  # Switch to channel-first
    mean, std = np.array(MEAN), np.array(STD)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0)


def apply_op(image, op, severity, image_size):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity, image_size)
    return np.asarray(pil_img) / 255.


def augmix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: Raw input image as float32 np.ndarray of shape (h, w, c)
        severity: Severity of underlying augmentation operators (between 1 to 10).
        width: Width of augmentation chain. Number of augmentation chains to mix per augmented example
        depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]. Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]
        alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
        mixed: Augmented and mixed image.
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image)
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity,
                                 [image.shape[1], image.shape[0]])
        # Preprocessing commutes since all coefficients are convex
        # mix += ws[i] * normalize(image_aug)
        mix += ws[i] * image_aug

    # mixed = (1 - m) * normalize(image) + m * mix
    mixed = (1 - m) * image + m * mix
    return mixed


def _get_pixels(per_pixel,
                rand_color,
                patch_size,
                dtype=torch.float32,
                device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype,
                           device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

class mixing_erasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels with different mixing operation.
    normal: original random erasing;
    soft: mixing ori with random pixel;
    self: mixing ori with other_ori_patch;
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465),
                 mode='pixel',
                 device='cpu',
                 type='normal',
                 mixing_coeff=[1.0, 1.0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.rand_color = False
        self.per_pixel = False
        self.mode = mode
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device
        self.type = type
        self.mixing_coeff = mixing_coeff

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if self.type == 'normal':
                    m = 1.0
                else:
                    m = np.float32(
                        np.random.beta(self.mixing_coeff[0],
                                       self.mixing_coeff[1]))
                if self.type == 'self':
                    x2 = random.randint(0, img.size()[1] - h)
                    y2 = random.randint(0, img.size()[2] - w)
                    img[:, x1:x1 + h,
                        y1:y1 + w] = (1 - m) * img[:, x1:x1 + h, y1:y1 +
                                                   w] + m * img[:, x2:x2 + h,
                                                                y2:y2 + w]
                else:
                    if self.mode == 'const':
                        img[0, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[0, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[0]
                        img[1, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[1, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[1]
                        img[2, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[2, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[2]
                    else:
                        img[:, x1:x1 + h, y1:y1 +
                            w] = (1 - m) * img[:, x1:x1 + h,
                                               y1:y1 + w] + m * _get_pixels(
                                                   self.per_pixel,
                                                   self.rand_color,
                                                   (img.size()[0], h, w),
                                                   dtype=img.dtype,
                                                   device=self.device)
                return img
        return img