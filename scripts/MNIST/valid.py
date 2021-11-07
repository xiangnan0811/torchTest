import os

import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch import nn, load, no_grad
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from model import MNISTModel


def valid_succeed():
    # 实例化模型
    model = MNISTModel()
    # 加载模型
    best_model_path = './models/mnist_best_model.pkl'
    if os.path.exists(best_model_path):
        model.load_state_dict(load(best_model_path))
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 加载数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    valid_set = MNIST(root='../../data', train=False, download=True, transform=transform)
    valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True)
    # 加入进度条
    valid_loader = tqdm(valid_loader, total=len(valid_loader))
    model.eval()
    succeed = []

    # 测试模型
    with no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            output = model(data)
            result = output.max(dim=1).indices
            result_mean = result.eq(target).float().mean()
            succeed.append(result_mean.item())
            # print(f'result:\n\t{result}')
            # print(f'target:\n\t{target}')
            # print(f'result_mean:\n\t{result_mean}')
            # loss = criterion(output, target)
            # valid_loader.set_description(f'Loss: {loss.item():.4f}')


    # 计算准确率
    accuracy = np.mean(succeed)
    return accuracy
