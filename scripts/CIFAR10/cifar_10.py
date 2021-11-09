# encoding=utf-8
import os
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
from torch import nn as nn
from torch import optim as optim

from model import CIFAR10Model, AdaptiveAvgCNNNet, LeNet, VGG 
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


def multi_check_accuracy(models, loader, device):
    total_correct = []
    model_correct = [[] for _ in range(len(models))]
    loader = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            for index, model in enumerate(models):
                model.eval()
                out = model(img)
                result = out.max(dim=1).indices
                result_mean = result.eq(label).float().mean()
                model_correct[index].append(result_mean.item())

    for index, model in enumerate(models):
        accuracy = np.mean(model_correct[index])
        print(f'{model.__class__.__name__} 模型的准确率为：{accuracy*100:.2f}%')
        total_correct.append(accuracy)

    return np.mean(total_correct)


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
            loss_value = loss.item()
            train_loader.set_description(f'loss: {loss_value:.4f}')
            epoch_loss.append(loss_value)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

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


def multi_train(epochs, models, train_loader, valid_loader, optimizer, criterion, device):
    best_accuracy = 0
    best_models = [None for _ in range(len(models))]
    for epoch in range(1, epochs+1):
        # 动态修改参数学习率
        if epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] *= 0.1
        # epoch_loss
        epoch_loss = []
        # 训练模型
        train_loader = tqdm(train_loader, total=len(train_loader))
        for images, labels in train_loader:
            # 指定运行环境
            images, labels = images.to(device), labels.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 多模型
            for model in models:
                model.train()
                out = model(images)
                model_loss = criterion(out, labels)
                epoch_loss.append(model_loss.item())
                model_loss.backward()
            # 更新参数
            optimizer.step()

        epoch_accuracy = multi_check_accuracy(models, valid_loader, device)
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            for index, model in enumerate(models):
                best_models[index] = deepcopy(model)
        print(f'\n多模型集成第{epoch}个epoch训练完成, 损失为 {np.mean(epoch_loss):.4f}, 准确率为 {epoch_accuracy*100:.2f}%, 当前最高准确率为 {best_accuracy*100:.2f}%\n')

    return best_models


def main():
    # 超参数
    batch_size = 64
    learning_rate = 0.01
    epochs = 10
    momentum = 0.9
    num_train = 49000
    device = specify_device(use_gpu=True)

    # 实例化模型 定义损失函数 定义优化器
    model = VGG('VGG16')
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


def multi_main():
    # 超参数
    batch_size = 64
    learning_rate = 0.001
    epochs = 50
    num_train = 49000
    device = specify_device(use_gpu=True)

    # 模型集成
    models = [CIFAR10Model().to(device), AdaptiveAvgCNNNet().to(device), LeNet().to(device)]
    # 实例化模型 定义损失函数 定义优化器
    criterion = nn.CrossEntropyLoss()
    optimizer= optim.Adam([{"params":model.parameters()} for model in models],lr=learning_rate)

    # 数据准备
    train_loader = prepare_data('train', batch_size, num_train)
    valid_loader = prepare_data('valid', batch_size, num_train)
    test_loader = prepare_data('test', batch_size)

    # 训练
    best_models = multi_train(epochs, models, train_loader, valid_loader, optimizer, criterion, device)
    print(f'\n{"-" * 30} 多模型集成训练完成 {"-" * 30}\n')
    # 测试
    test_best_accuracy = multi_check_accuracy(best_models, test_loader, device)
    print(f'多模型集成最优模型最终测试准确率为：{test_best_accuracy*100:.2f}%')


if __name__ == '__main__':
    # 普通单模型
    main()
    # 多模型集成
    # multi_main()
