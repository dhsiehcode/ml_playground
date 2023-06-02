import math
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from scipy.spatial import distance
import matplotlib.pyplot as plt

import pca_tests

### Variables ###

img_path = "C:\Dennis\Personal\Projects\ml_playground\data\VIP_2022_fall_classification\Train"
dirs = os.listdir(img_path)
show_dt = False
X = []
Y = []
num_imgs = 1000
n_components = 150
labeled_img_path = "C:\Dennis\Personal\Projects\ml_playground\data\VIP_2022_fall_classification\Train"
num_neighbors = int(math.sqrt(num_imgs)) #use rule of k = n ^ 1/2
dist_metric = distance.mahalanobis()

X, Y = pca_tests.pca_by_class(num_imgs, labeled_img_path, n_components)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=23)

knn = KNeighborsClassifier(n_neighbors= num_neighbors, weights='distance', metric='mahalanobis')

knn.fit(X_train,Y_train)

knn.score(X_test, Y_test)




