import os
import numpy as np
from PIL import Image
import cv2



def combine_train(labeled_img_path):
    classes = os.listdir(labeled_img_path)
    combined_path = []

    for c in classes:
        combined_path.append(os.listdir(os.path.join(labeled_img_path, c)))

    return np.asarray(combined_path)

def img_to_numpy_arr(imgs_path, num_imgs):

    labels = []

    for f in os.listdir(imgs_path[:num_imgs]):
        labels.append(np.array(Image.open(f)))

    return labels

def img_dir_to_numpy_arr(labeled_img_path, num_imgs):

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
    output = []
    for img in imgs:
        output.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    return np.asarray(output)
