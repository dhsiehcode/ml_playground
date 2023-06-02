import os
from PIL import Image
import numpy as np

from sklearn import tree
from sklearn import model_selection
import matplotlib.pyplot as plt

import pca_tests

img_path = "C:\Dennis\Personal\Projects\ml_playground\data\VIP_2022_fall_classification\Train"

dirs = os.listdir(img_path)

show_dt = False

X = []
Y = []
num_imgs = 1000
labeled_img_path = "C:\Dennis\Personal\Projects\ml_playground\data\VIP_2022_fall_classification\Train"
n_components = 200

## run PCA

X, Y = pca_tests.pca_by_class(num_imgs, labeled_img_path, n_components)


# for working with directories


#assert(len(X) == len(Y))

X = np.asarray(X)
Y = np.asarray(Y)

#print(X.shape)
#print(Y.shape)



X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=23)

#print(X_train.shape)
#print(Y_train.shape)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)

#print(X_train.shape[1])
#print(dirs)

if show_dt:

    fig = plt.figure(figsize=(55,50))
    tree.plot_tree(clf,
                       feature_names= range(X_train.shape[1]),
                       class_names= dirs,
                       filled=True)

    plt.show()

    fig.savefig("decision_tree.png")



print(clf.score(X_test, Y_test))

