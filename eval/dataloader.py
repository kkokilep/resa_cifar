import torch
from torchvision.datasets import CIFAR10 as C10
from torchvision.datasets import CIFAR100 as C100
import torchvision.transforms as transforms
from abc import ABCMeta
from functools import lru_cache
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import medmnist
from medmnist import INFO, Evaluator

@torch.no_grad()
def get_data(model, loader, output_size, device):
    """ encodes whole dataset into embeddings """
    xs = torch.empty(
        len(loader), loader.batch_size, output_size, dtype=torch.float32, device=device
    )
    ys = torch.empty(len(loader), loader.batch_size, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.cuda()
            xs[i] = model(x).to(device)
            y = y.squeeze()
            ys[i] = y.to(device)
    xs = xs.view(-1, output_size)
    ys = ys.view(-1)
    return xs, ys

class BaseDataset(metaclass=ABCMeta):
    """
        base class for datasets, it includes 2 types:
            - for classifier training for evaluation,
            - for testing
    """
    def __init__(
        self, args, bs_clf=1000, bs_test=1,
    ):
        self.bs_clf, self.bs_test = bs_clf, bs_test
        self.workers = args.workers
        self.data_path = args.data_path
        self.data_path_train = args.data_path_train
        self.data_path_test = args.data_path_test
        self.dataset_type = args.dataset
    
    @property
    @lru_cache()
    def clf(self):
        return DataLoader(
            dataset=self.ds_clf(),
            batch_size=self.bs_clf,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    @property
    @lru_cache()
    def test(self):
        return DataLoader(
            dataset=self.ds_test(),
            batch_size=self.bs_test,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
        )

class CIFAR10_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    def ds_clf(self):
        return C10(root=self.data_path, train=True, download=True, 
                transform=self.base_transform())

    def ds_test(self):
        return C10(root=self.data_path, train=False, download=True, 
                transform=self.base_transform())
    


class TinyImageNet200_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    def ds_clf(self):
        return datasets.ImageFolder(root=self.data_path_train, transform=self.base_transform())

    def ds_test(self):
        return datasets.ImageFolder(root=self.data_path_test, transform=self.base_transform())
    

class STL10_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    def ds_clf(self):
        return datasets.STL10(
            self.data_path_train,
            split="train",
            download=True,
            transform=self.base_transform(),
        )

    def ds_test(self):
        return datasets.STL10(
            self.data_path_test,
            split="test",
            download=True,
            transform=self.base_transform(),
        )
    

class Cinic10_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    def ds_clf(self):
        return datasets.ImageFolder(root=self.data_path_train, transform=self.base_transform())

    def ds_test(self):
        return datasets.ImageFolder(root=self.data_path_test, transform=self.base_transform())
    

class BloodMNIST_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    def ds_clf(self):
        info = INFO['bloodmnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='train', transform=self.base_transform(), root=self.data_path_train,download=True)
        return dataset

    def ds_test(self):
        info = INFO['bloodmnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', transform=self.base_transform(), root=self.data_path_test,download=True)
        return dataset


class OrganAMNIST_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    def ds_clf(self):
        info = INFO['organamnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='train', transform=self.base_transform(), root=self.data_path_train)
        return dataset

    def ds_test(self):
        info = INFO['organamnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', transform=self.base_transform(), root=self.data_path_test)
        return dataset

class OrganSMNIST_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    def ds_clf(self):
        info = INFO['organsmnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='train', transform=self.base_transform(), root=self.data_path_train)
        return dataset

    def ds_test(self):
        info = INFO['organsmnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', transform=self.base_transform(), root=self.data_path_test)
        return dataset
    

class OrganCMNIST_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    def ds_clf(self):
        info = INFO['organcmnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='train', transform=self.base_transform(), root=self.data_path_train)
        return dataset

    def ds_test(self):
        info = INFO['organcmnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', transform=self.base_transform(), root=self.data_path_test)
        return dataset

class CIFAR100_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    def ds_clf(self):
        return C100(root=self.data_path, train=True, download=True, 
                    transform=self.base_transform())

    def ds_test(self):
        return C100(root=self.data_path, train=False, download=True, 
                    transform=self.base_transform())

class CIFAR100Noise_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    def ds_clf(self):
        return datasets.ImageFolder(root=self.data_path_train, transform=self.base_transform())

    def ds_test(self):
        return datasets.ImageFolder(root=self.data_path_test, transform=self.base_transform())

def get_clf(args):
    if args.dataset == 'cifar10':
        return CIFAR10_clf(args)
    elif args.dataset == 'tinyimagenet200':
        return TinyImageNet200_clf(args)
    elif args.dataset == 'stl10':
        return STL10_clf(args)
    elif args.dataset == 'cinic10':
        return Cinic10_clf(args)
    elif 'noise' in args.dataset:
        return CIFAR100Noise_clf(args)
    elif args.dataset == 'bloodmnist':
        return BloodMNIST_clf(args)
    elif args.dataset == 'organamnist':
        return OrganAMNIST_clf(args)
    elif args.dataset == 'organsmnist':
        return OrganSMNIST_clf(args)
    elif args.dataset == 'organcmnist':
        return OrganCMNIST_clf(args)
    elif args.dataset == 'cifar100' or args.dataset == 'cifar100_noise':
        return CIFAR100_clf(args)
