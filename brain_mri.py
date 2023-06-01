import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA, KernelPCA
import cv2
from PIL import Image


import tools

def get_images(p):

    x = []
    down_widths = 70
    down_length = 70
    files = os.listdir(p)

    for f in files:
        # filter out image of incorrect size
        img = Image.open(os.path.join(p, f))
        resized_img = img.resize((down_widths, down_length))

        img_arr = np.array(resized_img)

        if img_arr.ndim == 3:
            img_arr = np.array(cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY))
        x.append(img_arr)

    return np.asarray(x)


if __name__ == '__main__':

    yes_path = 'C:\Dennis\Personal\Projects\ml_playground\data\\brain_mri\yes'
    no_path = 'C:\Dennis\Personal\Projects\ml_playground\data\\brain_mri\\no'

    yes_imgs = get_images(yes_path)
    no_imgs = get_images(no_path)


    #yes_imgs = tools.rgb_to_grayscale(yes_imgs)
    #no_imgs = tools.rgb_to_grayscale(no_imgs)

    yes_imgs = tools.flatten_imgs(yes_imgs)
    no_imgs = tools.flatten_imgs(no_imgs)




    pca = PCA(n_components=40)
    pca.fit(yes_imgs)
    yes_variance = pca.explained_variance_ratio_
    pca.fit(no_imgs)
    no_variance = pca.explained_variance_ratio_

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('Scree Plot Yes')
    ax2.title.set_text('Scree Plot No')
    ax1.plot(np.arange(0, 40), yes_variance)
    ax2.plot(np.arange(0, 40), no_variance)

    plt.show()




