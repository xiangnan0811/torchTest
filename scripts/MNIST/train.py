# encoding=utf-8
import os

import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn, optim, device, cuda, save, load

import valid
from model import MNISTModel

# 选取设备
device = device('cuda' if cuda.is_available() else 'cpu')
# 实例化模型
model = MNISTModel()
# 将模型放入设备
model = model.to(device)
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 加载已经训练好的模型和优化器继续进行训练
best_model_path = './models/mnist_best_model.pkl'
last_model_path = './models/mnist_last_model.pkl'
best_optimizer_path = './models/mnist_best_optimizer.pkl'
last_optimizer_path = './models/mnist_last_optimizer.pkl'
if os.path.exists(last_model_path):
    model.load_state_dict(load(last_model_path))
    optimizer.load_state_dict(load(last_optimizer_path))
# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_set = MNIST(root='../../data', train=True, download=True, transform=transform)


def train(epoch):
    # 动态修改参数学习率
    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.1
    # total_loss
    epoch_loss = []
    # 加入进度条
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    train_loader = tqdm(train_loader, total=len(train_loader))
    model.train()

    # 训练模型
    for images, labels in train_loader:
        # 将模型放入设备
        images = images.to(device)
        labels = labels.to(device)
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

    save(model.state_dict(), last_model_path)
    save(optimizer.state_dict(), last_optimizer_path)
    mean_loss = np.mean(epoch_loss)
    return mean_loss, model, optimizer


def save_model_and_optimizer(model, optimizer, file_path):
    save(model.state_dict(), file_path)
    save(optimizer.state_dict(), file_path)


def main():
    high_accuracy = 0
    epochs = 10
    for epoch in range(1, epochs+1):
        epoch_loss, model, optimizer = train(epoch)
        epoch_accuracy = valid.valid_succeed()
        if epoch_accuracy > high_accuracy:
            high_accuracy = epoch_accuracy
            # 保存最优模型
            save(model.state_dict(), best_model_path)
            # 保存最优参数
            save(optimizer.state_dict(), best_optimizer_path)
        print(f'第{epoch}个epoch训练完成, 损失为 {epoch_loss:.4f}, 准确率为 {epoch_accuracy:.4f}, 当前最高准确率为 {high_accuracy:.4f}')
    print(f'{"-" * 30} 训练完成 {"-" * 30}')


if __name__ == '__main__':
    main()
