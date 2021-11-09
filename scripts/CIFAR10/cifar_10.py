# encoding=utf-8
import os

from tqdm import tqdm
import numpy as np
import torch
from torch import nn as nn
from torch import optim as optim

from model import CIFAR10Model
from utils import prepare_data, specify_device


def check_accuracy(model, loader, device):
    model.eval()
    succeed = []
    loader = tqdm(loader, total=len(loader))
    # 检测模型准确率
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            result = output.max(dim=1).indices
            result_mean = result.eq(target).float().mean()
            succeed.append(result_mean.item())
    # 计算准确率
    accuracy = np.mean(succeed)
    return accuracy


def train(epochs, model, train_loader, valid_loader, optimizer, criterion, device):
    model = model.to(device)
    best_accuracy = 0
    for epoch in range(1, epochs+1):
        # 动态修改参数学习率
        if epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] *= 0.1
        # epoch_loss
        epoch_loss = []
        # 训练模型
        train_loader = tqdm(train_loader, total=len(train_loader))
        for images, labels in train_loader:
            model.train()
            # 指定运行环境
            images, labels = images.to(device), labels.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 预测
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            epoch_loss.append(loss.item())
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # # 打印损失
            # train_loader.set_description(f'loss: {loss.item():.4f}')

        epoch_accuracy = check_accuracy(model, valid_loader, device)
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            # 保存最优模型
            torch.save(model.state_dict(), './models/cifar10_best_model.pkl')
            # 保存最优参数
            torch.save(optimizer.state_dict(), './models/cifar10_best_optimizer.pkl')
        print(f'\n第{epoch}个epoch训练完成, 损失为 {np.mean(epoch_loss):.4f}, 准确率为 {epoch_accuracy*100:.2f}%, 当前最高准确率为 {best_accuracy*100:.2f}%\n')

    model.load_state_dict(torch.load('./models/cifar10_best_model.pkl'))
    return model


def main():
    # 超参数
    batch_size = 64
    learning_rate = 0.01
    epochs = 10
    momentum = 0.9
    num_train = 49000
    device = specify_device(use_gpu=True)

    # 实例化模型 定义损失函数 定义优化器
    model = CIFAR10Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # 数据准备
    train_loader = prepare_data('train', batch_size, num_train)
    valid_loader = prepare_data('valid', batch_size, num_train)
    test_loader = prepare_data('test', batch_size)

    # 训练
    best_model = train(epochs, model, train_loader, valid_loader, optimizer, criterion, device)
    print(f'\n{"-" * 30} 训练完成 {"-" * 30}\n')
    # 测试
    test_best_accuracy = check_accuracy(best_model, test_loader, device)
    print(f'最优模型测试准确率为：{test_best_accuracy*100:.2f}%')


if __name__ == '__main__':
    main()
