import os
from PIL import Image
import numpy as np

from sklearn import tree
from sklearn import model_selection
import matplotlib.pyplot as plt

img_path = "C:\Dennis\Personal\Projects\ml_playground\data\VIP_2022_fall_classification\Train"

dirs = os.listdir(img_path)

X = []
Y = []
num_imgs = 1000

for dir in dirs:
    p = os.path.join(img_path, dir)
    files = os.listdir(p)
    img_count = 0
    for f in files:
        if f.endswith('.jpg'):
            img = np.array((Image.open(os.path.join(p, f))))
            if img.shape == (150, 150, 3):
                X.append(img)
                Y.append(dir)
                img_count += 1
                if img_count >= num_imgs:
                    break

for i in range(len(X)):
    X[i] = X[i].reshape(-1)

X = np.asarray(X)
Y = np.asarray(Y)

#print(X.shape)
#print(Y.shape)


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=23)

#print(X_train.shape)
#print(Y_train.shape)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)


fig = plt.figure(figsize=(55,50))
_ = tree.plot_tree(clf,
                   feature_names= X_train.shape[1],
                   class_names= dirs,
                   filled=True)



print(clf.score(X_test, Y_test))

