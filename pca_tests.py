import os
import cv2
import tools
import numpy as np
from sklearn.decomposition import PCA

labeled_img_path = "C:\Dennis\Personal\Projects\ml_playground\data\VIP_2022_fall_classification\Train"
num_imgs = 100
n_components = 5

train_labels = tools.img_dir_to_numpy_arr(labeled_img_path, num_imgs)

files = os.listdir(labeled_img_path)

img_arr = {}

for i in range(len(files)):
    img_arr[files[i]] = train_labels[i]

    train_labels[i] = tools.rgb_to_grayscale(train_labels[i])



    pca = PCA(n_components)
    new_val = pca.fit_transform(np.asarray(train_labels[i]))

    cv2.imshow(f'image{i}', new_val[0])
    cv2.waitKey(0)





