import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

import tools


'''

'''
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


def show_scree_plot(yes_imgs, no_imgs, max_components = 40):




    pca = PCA(n_components=max_components)
    pca.fit(yes_imgs)
    yes_variance = pca.explained_variance_ratio_
    pca.fit(no_imgs)
    no_variance = pca.explained_variance_ratio_

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('Scree Plot Yes')
    ax2.title.set_text('Scree Plot No')
    ax1.plot(np.arange(0, max_components), yes_variance)
    ax2.plot(np.arange(0, max_components), no_variance)

    plt.show()

def show_pca_diff(yes_imgs, no_imgs, components = 20):

    # also compares before and after

    #yes images first
    yes_before = yes_imgs[0].reshape(70, 70)
    pca = PCA(n_components=components)
    yes_x = pca.fit_transform(yes_imgs)
    yes_after = pca.inverse_transform(yes_x)[0].reshape(70, 70)

    no_before = no_imgs[0].reshape(70, 70)
    pca = PCA(n_components=components)
    no_x = pca.fit_transform(no_imgs)
    no_after = pca.inverse_transform(no_x)[0].reshape(70, 70)


    #fig = plt.figure()
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(yes_before)
    axs[0, 0].set_title('Yes Before')
    axs[0, 1].imshow(yes_after)
    axs[0, 1].set_title('Yes After')
    axs[1, 0].imshow(no_before)
    axs[1, 0].set_title('No Before')
    axs[1, 1].imshow(no_after)
    axs[1, 1].set_title('No After')
    plt.show()

    return yes_x, no_x

def pca(yes_imgs, no_imgs, components = 20):

    pca = PCA(n_components=components)
    yes_x = pca.fit_transform(yes_imgs)

    pca = PCA(n_components=components)
    no_x = pca.fit_transform(no_imgs)

    return yes_x, no_x

def knn(yes_imgs, no_imgs):

    yes_label = np.zeros(len(yes_imgs))
    no_label = np.ones(len(no_imgs))

    final_label = np.concatenate((yes_label, no_label))
    final_imgs = np.concatenate((yes_imgs, no_imgs))

    assert(final_label.shape[0] == final_imgs.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(final_imgs, final_label, test_size = 0.15, random_state = 42)

    knn = KNeighborsClassifier(n_neighbors=int(math.sqrt(X_train.shape[0])))

    knn.fit(X_train, y_train)

    print(knn.score(X_test, y_test))




if __name__ == '__main__':


    yes_path = 'C:\Dennis\Personal\Projects\ml_playground\data\\brain_mri\yes'
    no_path = 'C:\Dennis\Personal\Projects\ml_playground\data\\brain_mri\\no'

    yes_imgs = get_images(yes_path)
    no_imgs = get_images(no_path)

    yes_imgs = tools.flatten_imgs(yes_imgs)
    no_imgs = tools.flatten_imgs(no_imgs)

    #show_pca_diff(yes_imgs, no_imgs, 40)
    #show_scree_plot()

    knn(yes_imgs, no_imgs)









