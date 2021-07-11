import numpy as np
import cv2
import os
import argparse
import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mask')
    parser.add_argument('src')
    parser.add_argument('ref')
    parser.add_argument('output')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    mask_files = os.listdir(args.mask)
    src_files = os.listdir(args.src)

    rmse = []
    for m, s in zip(mask_files, src_files):
        im_mask = cv2.imread('{}/{}'.format(args.mask, m))
        mask = (im_mask[:, :, 0] < 255) & (im_mask[:, :, 1] < 255) & (im_mask[:, :, 2] < 255)
        idx = int(m.split('.')[0])

        im_src = cv2.cvtColor(cv2.imread('{}/{}'.format(args.src, s)), cv2.COLOR_BGR2GRAY)
        im_ref = cv2.cvtColor(cv2.imread('{}/frame-{:06d}.color.png'.format(args.ref, idx)), cv2.COLOR_BGR2GRAY)

        diff = mask * ((im_src - im_ref) / 255.0)
        diff = diff ** 2

        plt.imsave('{}/{:03d}.png'.format(args.output, idx), diff)

        result = np.sqrt(np.mean(diff[mask]))
        rmse.append(result)

        print(result)

    print(np.mean(rmse))
    print(np.std(rmse))

