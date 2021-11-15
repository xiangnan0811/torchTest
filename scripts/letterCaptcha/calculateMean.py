from tqdm import tqdm
import numpy as np
import os
import cv2
from torchvision import transforms as T
from letterCaptchaDataset import LetterCaptchaDataset


def calculate_mean_and_std(files_dir):
    transform = T.Compose([
        T.ToTensor(),
    ])
    captchas = LetterCaptchaDataset(files_dir, transform)

    R_channel, G_channel, B_channel = 0., 0., 0.
    R_channel_square, G_channel_square, B_channel_square = 0., 0., 0.
    pixels_num = 0

    captchas = tqdm(captchas, total=len(captchas))

    for image, _ in captchas:
        img = image.permute(1, 2, 0).numpy()
        h, w, _ = img.shape
        pixels_num += h*w

        R_temp = img[:, :, 0]
        R_channel += np.sum(R_temp)
        R_channel_square += np.sum(np.power(R_temp, 2.0))

        G_temp = img[:, :, 1]
        G_channel += np.sum(G_temp)
        G_channel_square += np.sum(np.power(R_temp, 2.0))

        B_temp = img[:, :, 2]
        B_channel += np.sum(B_temp)
        B_channel_square += np.sum(np.power(R_temp, 2.0))

    R_mean, G_mean, B_mean = R_channel/pixels_num, G_channel/pixels_num, B_channel/pixels_num

    R_std = np.sqrt(R_channel_square/pixels_num - R_mean*R_mean)
    G_std = np.sqrt(G_channel_square/pixels_num - G_mean*G_mean)
    B_std = np.sqrt(B_channel_square/pixels_num - B_mean*B_mean)

    print(f'({R_mean:.4f}, {G_mean:.4f}, {B_mean:.4f}), ({B_std:.4f}, {R_std:.4f}, {G_std:.4f})')

if __name__ == '__main__':
    calculate_mean_and_std(files_dir='D:/code/deepLearning/torch/torchTest/data/letter_captcha/valid/')
