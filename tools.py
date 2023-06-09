import os
import pandas as pd
import numpy
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def import_data(path):

    df = pd.read_csv(path)

    col = df.columns


    return df

def show_compare_imgs(imgs):
    '''

    shows images to be compared with each other

    :param imgs:
    :return:
    '''

    n_images = len(imgs[0])

    if n_images >= 4:
        n_images = 4

    rows = len(imgs)

    fig, axs = plt.subplots(rows, n_images)

    for row in range(rows):
        for i in range(n_images):
            img = imgs[row][i].reshape((150,150))
            axs[row,i].imshow(img, cmap='gray')

    #plt.tight_layout()
    plt.show()

def show_imgs(imgs):
    '''

    displays the images specified by imgs

    :param
        imgs: list of images
    '''

    n_images = len(imgs)

    # if it is an array
    if isinstance(imgs[0], np.ndarray):
        if n_images >= 4:
            n_images = 4
        fig, ax = plt.subplots(1, n_images)

        for i in range(n_images):

            img = imgs[i].reshape((150,150))
            ax[i].imshow(img, cmap='gray')

        plt.tight_layout()
        plt.show()

    else:
        if n_images > 5:
            n_images = 5
        fig, axs = plt.subplots(1, n_images)

        for i in range(n_images):
            img = np.array(Image.open(imgs[i])).reshape((150,150))
            axs[0, i].imshow(img, cmap='gray')

        plt.tight_layout()
        plt.show()



def flatten_imgs(imgs):

    '''

    flattens all the images given to be 1D

    :param
        imgs: list of 1 x N images where each element is a D x D image
    :return
        output: 2D array of 1 x N flattened images
    '''

    output = []
    for img in imgs:
        output.append(img.flatten())

    return np.asarray(output)

def combine_train(labeled_img_path):
    '''

    combineds all images in a directory to one array with the path to all images

    :param
        labeled_img_path: path to training file where each subdirectory is a class
    :return:
        combined_path: an array where each element is the path to an image
    '''
    classes = os.listdir(labeled_img_path)
    combined_path = []

    for c in classes:
        combined_path.append(os.listdir(os.path.join(labeled_img_path, c)))

    return np.asarray(combined_path)

def img_to_numpy_arr(imgs_path, num_imgs = 0, required_shape = (0,0,0)):

    '''

    converts all images specified into numpy arrays

    :param
        imgs_path: the path for the images
        num_imgs: number of images
    :return:
        labels: list of arrays where each element in an array representing an image
    '''
    labels = []
    files = os.listdir(imgs_path)
    if num_imgs != 0:
        files = files[:num_imgs]


    if required_shape != (0,0,0):


        for f in files:

            # filter out image of incorrect size
            img = np.array(Image.open(os.path.join(imgs_path, f)))
            if img.shape == required_shape:
                labels.append(img)

    else:
        for f in files:
            # filter out image of incorrect size
            img = np.array(Image.open(os.path.join(imgs_path, f)))
            labels.append(img)

    return np.asarray(labels)

def img_dir_to_numpy_arr(labeled_img_path, num_imgs):

    '''

    converts images from a directory into numpy arrays

    :param
        labeled_img_path: path to training directory where each subdirectory is a class
        num_imgs: the number of  images to convert to numpy array
    :return:
        labels: list of images base on class. Each element of the list has
                images from a class. This means each element is a list of arrays where each array represents an image

    '''

    images = []
    labels = []

    for name in os.listdir(labeled_img_path):
        class_path = os.path.join(labeled_img_path, name)

        files = os.listdir(class_path)
        #files = files[:num_imgs]
        #print(files)
        os.chdir(class_path)
        image = []
        img_count = 0
        for i in range(len(files)):
            img = np.array((Image.open(files[i])))
            if img.shape == (150, 150, 3):
                image.append(img)
                labels.append(name)
                img_count += 1
                if img_count >= num_imgs:
                    break
        #label = np.array([np.array(Image.open(fname)) for fname in files])
        images.append(image)

    return images, labels


def rgb_to_grayscale(imgs):
    '''

    converts images from rgb to grayscale

    :param
        imgs: the list of images to be converted to rgb
    :return:
        output: a list of images converted to grayscale
    '''
    output = []
    for img in imgs:
        output.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    return np.asarray(output)
