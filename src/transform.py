# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import random
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as transforms
import torch
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import random
import medmnist
from medmnist import INFO, Evaluator

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.
        Args:
            img (Image): an image in the PIL.Image format.
        Returns:
            Image: solarized image.
        """
        return ImageOps.solarize(img)

class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)

class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)


class MultiVisionDataset(datasets.VisionDataset):
    def __init__(
        self,
        data_path,
        args,
        dataset_type='cifar10',  # CIFAR10 or CIFAR100 or STL10
        download=True,
        return_index=False,
    ):
        # Load the appropriate dataset
        if dataset_type == 'cifar10':
            self.dataset = datasets.CIFAR10(data_path, train=True, download=download)
            self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif dataset_type == 'cifar100':
            self.dataset = datasets.CIFAR100(data_path, train=True, download=download)
            self.mean, self.std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        elif dataset_type == 'organamnist':
            self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            info = INFO[dataset_type]
            DataClass = getattr(medmnist, info['python_class'])
            self.dataset = DataClass(split='train', transform=None, root=data_path)
        elif dataset_type == 'organsmnist':
            self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            info = INFO[dataset_type]
            DataClass = getattr(medmnist, info['python_class'])
            self.dataset = DataClass(split='train', transform=None, root=data_path)
        elif dataset_type == 'organcmnist':
            self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            info = INFO[dataset_type]
            DataClass = getattr(medmnist, info['python_class'])
            self.dataset = DataClass(split='train', transform=None, root=data_path)
        elif dataset_type == 'bloodmnist':
            self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            info = INFO[dataset_type]
            DataClass = getattr(medmnist, info['python_class'])
            self.dataset = DataClass(split='train', transform=None, root=data_path)
        elif dataset_type == 'tinyimagenet200':
            self.dataset = datasets.ImageFolder(data_path, transform=None)
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif dataset_type == 'cifar100_noise_5_percent':
            self.dataset = datasets.ImageFolder(data_path, transform=None)
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif dataset_type == 'cinic10':
            self.dataset = datasets.ImageFolder(data_path, transform=None)
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif dataset_type == 'stl10':
            self.dataset = datasets.STL10(data_path, split="train+unlabeled", download=download,transform=None)
            self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        else:
            raise ValueError("Invalid dataset type.")
        
        self.dataset_type = dataset_type

        # Limit the dataset size if specified
        if args.size_dataset >= 0:
            self.dataset.data = self.dataset.data[:args.size_dataset]
            self.dataset.targets = self.dataset.targets[:args.size_dataset]

        if args.aug_strength == 'weak' and args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'bloodmnist' or args.dataset == 'cinic10':
            weak_transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

            self.return_index = return_index

            trans = [weak_transform]
            

            for i in range(len(args.crops_nmb)):
                trans.extend([VisionTransform(crop_size=32, solarization_prob=args.solarization_prob[i],
                                              mean=self.mean, std=self.std)] * args.crops_nmb[i])
        elif args.aug_strength == 'weak' and args.dataset == 'stl10':
            weak_transform = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

            self.return_index = return_index

            trans = [weak_transform]
            

            for i in range(len(args.crops_nmb)):
                trans.extend([VisionTransform(crop_size=96, solarization_prob=args.solarization_prob[i],
                                              mean=self.mean, std=self.std)] * args.crops_nmb[i])   
        
        elif args.aug_strength == 'weak' and args.dataset == 'tinyimagenet200':
            weak_transform = transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

            self.return_index = return_index

            trans = [weak_transform]
            

            for i in range(len(args.crops_nmb)):
                trans.extend([VisionTransform(crop_size=64, solarization_prob=args.solarization_prob[i],
                                              mean=self.mean, std=self.std)] * args.crops_nmb[i])
        elif args.aug_strength == 'weak' and 'organ' in args.dataset:
            weak_transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

            self.return_index = return_index

            trans = [weak_transform]
            

            for i in range(len(args.crops_nmb)):
                trans.extend([VisionTransformGray(crop_size=32, solarization_prob=args.solarization_prob[i],
                                              mean=self.mean, std=self.std)] * args.crops_nmb[i])
        
        elif args.aug_strength == 'asymmetric':
            trans = []
            self.return_index = return_index
            trans.append(VisionTransform(crop_size=args.crops_size[0],min_scale=0.08,gaussian_prob = 1.0, solarization_prob=0.0))
            trans.append(VisionTransform(crop_size=args.crops_size[0],min_scale=0.08,gaussian_prob = 0.1, solarization_prob=0.2))
        else:
            trans = []
            self.return_index = return_index
            for i in range(len(args.crops_nmb)+1):
                trans.extend([VisionTransform(crop_size=args.crops_size[0], solarization_prob=args.solarization_prob[0], 
                                            mean=self.mean, std=self.std) ])

            print(trans)

        self.trans = trans

    def __getitem__(self, index):
        # Fetch the image and target from the dataset
        if 'mnist' in self.dataset_type:
            image = self.dataset.imgs[index]
        elif 'noise' in self.dataset_type or 'cinic' in self.dataset_type or 'tinyimagenet' in self.dataset_type:
            image = self.dataset.samples[index][0]
        else:
            image = self.dataset.data[index]
        
        if self.dataset_type == 'stl10':
            image = Image.fromarray(np.transpose(image, (1, 2, 0)))
        elif 'noise' in self.dataset_type:
            image = Image.open(image).convert('RGB')
        else:
            image = Image.fromarray(image)

        # Apply the transformations
        crops = list(map(lambda trans: trans(image), self.trans))

        if self.return_index:
            return index, crops
        return crops
    
    def __len__(self):
        if 'mnist' in self.dataset_type:
            return len(self.dataset.imgs)
        elif 'noise' in self.dataset_type:
            return len(self.dataset.samples)
        elif 'cifar' in self.dataset_type or 'stl10' in self.dataset_type:
            return len(self.dataset.data)
        else:
            return len(self.dataset.samples)
    


class VisionTransform(BaseTransform):
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.0,
        solarization_prob: float = 0.0,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
        crop_size: int = 32,
        mean: tuple = (0.4914, 0.4822, 0.4465),
        std: tuple = (0.2470, 0.2435, 0.2616),
    ):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

class VisionTransformGray(BaseTransform):
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.0,
        solarization_prob: float = 0.0,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
        crop_size: int = 32,
        mean: tuple = (0.4914, 0.4822, 0.4465),
        std: tuple = (0.2470, 0.2435, 0.2616),
    ):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

