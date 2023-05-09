import os
import numpy as np
from PIL import Image
import cv2

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

def img_to_numpy_arr(imgs_path, num_imgs):

    '''

    converts all images specified into numpy arrays

    :param
        imgs_path: the path for the images
        num_imgs: number of images
    :return:
        labels: list of arrays where each element in an array representing an image
    '''
    labels = []

    for f in os.listdir(imgs_path[:num_imgs]):
        labels.append(np.array(Image.open(f)))

    return labels

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

    labels = []

    labeled_img_path = labeled_img_path
    num_imgs = num_imgs

    for name in os.listdir(labeled_img_path):
        class_path = os.path.join(labeled_img_path, name)

        files = os.listdir(class_path)
        files = files[:num_imgs]
        #print(files)
        os.chdir(class_path)
        label = []
        for fname in files:
            label.append((np.array(Image.open(fname))))
        #label = np.array([np.array(Image.open(fname)) for fname in files])
        labels.append(label)

    return labels


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
