from tqdm import tqdm
import numpy as np
import os
import cv2


def calculate_mean_and_std(files_dir):
    files = os.listdir(files_dir)

    R, G, B = 0., 0., 0.
    R_2, G_2, B_2 = 0., 0., 0.
    N = 0
    files = tqdm(files, total=len(files))

    for file in files:
        img = cv2.imread(files_dir+file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        h, w, c = img.shape
        N += h*w

        R_t = img[:, :, 0]
        R += np.sum(R_t)
        R_2 += np.sum(np.power(R_t, 2.0))

        G_t = img[:, :, 1]
        G += np.sum(G_t)
        G_2 += np.sum(np.power(R_t, 2.0))

        B_t = img[:, :, 2]
        B += np.sum(B_t)
        B_2 += np.sum(np.power(R_t, 2.0))

    R_mean, G_mean, B_mean = R/N, G/N, B/N

    R_std = np.sqrt(R_2/N - R_mean*R_mean)
    G_std = np.sqrt(G_2/N - G_mean*G_mean)
    B_std = np.sqrt(B_2/N - B_mean*B_mean)

    print("R_mean: %f, G_mean: %f, B_mean: %f" % (R_mean, G_mean, B_mean))
    print("R_std: %f, G_std: %f, B_std: %f" % (R_std, G_std, B_std))
    return (R_mean, G_mean, B_mean), (R_std, G_std, B_std)

if __name__ == '__main__':
    res = calculate_mean_and_std(files_dir='/Users/weibo/code/Python/torchTest/data/letter_captcha/test/')
    print(res)
