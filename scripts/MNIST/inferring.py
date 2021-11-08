import os

from torchvision import transforms
from torch import load, no_grad
from PIL import Image

from model import MNISTModel


# 实例化模型
model = MNISTModel()
# 加载模型
model.load_state_dict(load('./models/mnist_best_model.pkl'))
# 加载数据集
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

image = Image.open('scripts/MNIST/3.jpg')
image = transform(image)

model.eval()

# 推理
with no_grad():
    output = model(image)
    result = output.max(dim=1).indices
    print(result.item())
