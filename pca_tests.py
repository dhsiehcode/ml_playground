import os
import cv2
import tools
import numpy as np
from sklearn.decomposition import PCA, KernelPCA



def kernel_pca_by_class(num_imgs, labeled_img_path, n_components):

    '''

    :param num_imgs:
    :param labeled_img_path:
    :param n_components:
    :return:
    '''

    train_labels = tools.img_dir_to_numpy_arr(labeled_img_path, num_imgs)

    files = os.listdir(labeled_img_path)

    img_dict ={}

    kernel_pca_result = []

    for i in range(len(files)):
        img_dict[files[i]] = train_labels[i]
        train_labels[i] = tools.rgb_to_grayscale(train_labels[i])
        flattened_labels = tools.flatten_imgs((train_labels[i]))
        kernel_pca = KernelPCA(n_components=n_components, kernel="rbf", fit_inverse_transform=True)
        kernel_pca_result.append(flattened_labels)
        ##
        x_transformed = kernel_pca.fit_transform(flattened_labels)
        kernel_pca_result.append(kernel_pca.inverse_transform(x_transformed))

    #tools.show_compare_imgs(kernel_pca_result)
    return kernel_pca_result

def pca_by_class(num_imgs, labeled_img_path, n_components):

    train_images, train_labels = tools.img_dir_to_numpy_arr(labeled_img_path, num_imgs)



    #print(len(train_images))
    #print(len(train_images[0]))
    #print(print(train_images[0][0].shape))
    #print((train_labels))


    files = os.listdir(labeled_img_path)

    img_dict = {}

    pca_result = []

    ## Regular PCA on each class

    for i in range(len(files)):
        img_dict[files[i]] = train_images[i]
        train_images[i] = tools.rgb_to_grayscale(train_images[i])
        flattened_labels = tools.flatten_imgs(train_images[i])
        pca = PCA(n_components)
        pca.fit(flattened_labels)
        transformed_result = pca.transform(flattened_labels)
        pca_result.extend(transformed_result)
        #tools.show_compare_imgs([train_labels[i], pca.components_[:n_components]])

    return pca_result, train_labels


def one_class_pca(num_imgs, labeled_img_path, n_components, type):


    labeled_img_path = os.path.join(labeled_img_path, type)

    #print(labeled_img_path)

    train_labels = tools.img_to_numpy_arr(labeled_img_path, num_imgs)

    train_labels = tools.rgb_to_grayscale(train_labels)

    flattened_labels = tools.flatten_imgs(train_labels)

    kernel_pca = KernelPCA(n_components=n_components, kernel="rbf", fit_inverse_transform=True)

    x_transformed = kernel_pca.fit_transform(flattened_labels)
    #tools.show_imgs(kernel_pca.inverse_transform(x_transformed))
    return kernel_pca.inverse_transform(x_transformed)


if __name__ == '__main__':
    labeled_img_path = "C:\Dennis\Personal\Projects\ml_playground\data\VIP_2022_fall_classification\Train"
    #num_imgs = 1200
    #n_components = 250

    num_imgs = 100
    n_components = 10

    one_class_pca(num_imgs, labeled_img_path, n_components, type= 'buildings')

   #kernel_pca_by_class(num_imgs, labeled_img_path, n_components)
    #pca_by_class(num_imgs, labeled_img_path, n_components)






