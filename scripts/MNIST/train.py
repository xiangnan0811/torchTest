# encoding=utf-8
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim, device, cuda, save, load
from tqdm import tqdm
import valid
import numpy as np
import os


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out


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
model_path = './models/mnist_model.pkl'
optimizer_path = './models/mnist_optimizer.pkl'
if os.path.exists(model_path):
    model.load_state_dict(load(model_path))
    optimizer.load_state_dict(load(optimizer_path))
# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_set = MNIST(root='../../data', train=True, download=True, transform=transform)


def train(epoch):
    # total_loss
    total_loss = []
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    # 加入进度条
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
        total_loss.append(loss.item())
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # # 打印损失
        # train_loader.set_description(f'loss: {loss.item():.4f}')


    # 保存模型
    save(model.state_dict(), model_path)
    # 保存参数
    save(optimizer.state_dict(), optimizer_path)
    print(f'第{epoch}个epoch训练完成, 损失为 {np.mean(total_loss):.4f}, 准确率为 {valid.valid_succeed():.4f}')


if __name__ == '__main__':
    for epoch in range(1, 11):
        print('-' * 20)
        train(epoch)
    print('训练完成')
    print('-' * 20)
