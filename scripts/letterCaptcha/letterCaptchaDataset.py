import os
import random

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms


class LetterCaptchaDataset(Dataset):

    def __init__(self, root: str, transform=None):
        super(LetterCaptchaDataset, self).__init__()
        self.path = root
        self.transform = transform
        # TODO 可优化（例：可从外部传入）
        self.mapping = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.suffix = ['png', 'jpg', 'jpeg']

    def load_captcha_path(self):
        captcha_list = list(os.walk(self.path))[0][2]
        captcha_list = [i for i in captcha_list if i.split('.')[-1].lower() in self.suffix]
        # TODO 相关错误判断逻辑
        return captcha_list

    def __len__(self):
        return len(self.load_captcha_path())

    def __getitem__(self, index):
        load_captcha = self.load_captcha_path()
        image_path = load_captcha[index]
        image = Image.open(f'{self.path}/{image_path}')
        if self.transform:
            image = self.transform(image)
        label = [self.mapping.index(i) for i in image_path.split('_')[0]]
        label = torch.as_tensor(label, dtype=torch.int64)
        return image, label


if __name__ == '__main__':
    captcha_path = os.path.join(os.getcwd(), 'data/letter_captcha')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    captchas = LetterCaptchaDataset(captcha_path, transform)
    print(list(captchas))

