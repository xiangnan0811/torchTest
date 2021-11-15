# encoding=utf-8
import os
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from letterCaptchaDataset import LetterCaptchaDataset
from letterCaptchaModels import LetterCaptchaModel


def specify_device(use_gpu: bool = True):
    """
    指定训练环境，默认使用 GPU ，若没有 GPU 则使用 CPU
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def check_accuracy(model, loader, device, batch_size):
    model.eval()
    succeed = []
    loader = tqdm(loader, total=len(loader))
    # 检测模型准确率
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.view(batch_size*4, 8)
            target = target.view(-1)
            result = output.max(dim=1)[1]
            result_mean = result.eq(target).float().mean()
            succeed.append(result_mean.item())
    # 计算准确率
    accuracy = np.mean(succeed)
    return accuracy


def train(epochs, model, train_loader, valid_loader, optimizer, criterion, device, batch_size):
    model = model.to(device)
    best_accuracy = 0
    for epoch in range(1, epochs+1):
        # 动态修改参数学习率
        if epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] *= 0.1
        # epoch_loss
        epoch_loss = []
        # 训练模型
        model.train()
        train_loader = tqdm(train_loader, total=len(train_loader))
        for images, labels in train_loader:
            # 指定运行环境
            images, labels = images.to(device), labels.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 预测
            outputs = model(images)
            outputs = outputs.view(batch_size*4, 8)
            labels = labels.view(-1)
            # 计算损失
            loss = criterion(outputs, labels)
            loss_value = loss.item()
            epoch_loss.append(loss_value)
            train_loader.set_description(f'loss: {np.mean(epoch_loss):.4f}')
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

        epoch_accuracy = check_accuracy(model, valid_loader, device, batch_size)
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            # 保存最优模型
            torch.save(model.state_dict(), './models/captcha_best_model.pkl')
            # 保存最优参数
            torch.save(optimizer.state_dict(), './models/captcha_best_optimizer.pkl')
        print(f'\n第{epoch}个epoch训练完成, 损失为 {np.mean(epoch_loss):.4f}, 准确率为 {epoch_accuracy*100:.2f}%, 当前最高准确率为 {best_accuracy*100:.2f}%\n')

    model.load_state_dict(torch.load('./models/captcha_best_model.pkl'))
    return model


def prepare(batch_size):
    captcha_root_path = os.path.join(os.getcwd(), 'data/letter_captcha')
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.9478, 0.9467, 0.9474), (0.1911, 0.1891, 0.1946))
    ])
    # 数据
    train_dataset = LetterCaptchaDataset(root=f'{captcha_root_path}/train', transform=transforms)
    valid_dataset = LetterCaptchaDataset(root=f'{captcha_root_path}/valid', transform=transforms)
    test_dataset = LetterCaptchaDataset(root=f'{captcha_root_path}/test', transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, valid_loader, test_loader


def main():
    # 超参数
    batch_size = 8
    learning_rate = 0.01
    epochs = 10
    momentum = 0.9
    device = specify_device(use_gpu=True)

    # 实例化模型 定义损失函数 定义优化器
    model = LetterCaptchaModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 数据准备
    train_loader, valid_loader, test_loader = prepare(batch_size)

    # 训练
    best_model = train(epochs, model, train_loader, valid_loader, optimizer, criterion, device, batch_size)
    print(f'\n{"-" * 30} 训练完成 {"-" * 30}\n')
    # 测试
    test_best_accuracy = check_accuracy(best_model, test_loader, device, batch_size)
    print(f'最优模型测试准确率为：{test_best_accuracy*100:.2f}%')


if __name__ == '__main__':
    # 普通单模型
    main()
