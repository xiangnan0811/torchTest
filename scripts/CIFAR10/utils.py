import torch
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, sampler


def specify_device(use_gpu: bool = True):
    """
    指定训练环境，默认使用 GPU ，若没有 GPU 则使用 CPU
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def prepare_data(mode: str, batch_size: int, num_train: int = 49000):
    """
    数据预处理
    """
    # 加载数据集
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # 数据增强
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    data_dir = '../../data'

    if mode == 'train':
        dataset = CIFAR10(data_dir, train=True, download=False, transform=transform_train)
        dataloader = DataLoader(dataset, batch_size, sampler=sampler.SubsetRandomSampler(range(num_train)))
    elif mode == 'valid':
        dataset = CIFAR10(data_dir, train=True, download=False, transform=transform_test)
        dataloader = DataLoader(dataset, batch_size, sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))
    else:
        dataset = CIFAR10(data_dir, train=False, download=False, transform=transform_test)
        dataloader = DataLoader(dataset, batch_size)
    return dataloader
