import math
import cv2
from PIL import Image
import tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


def score_model(clf, X_test, y_test, save_name = None):

    print(f"Testing on {y_test.shape[0]} images. {np.sum(y_test[y_test == 1])} no images and {np.sum(y_test[y_test == 0])} yes images")

    predictions = clf.predict(X_test)

    # PR Curve
    precisions, recalls, threshold = metrics.precision_recall_curve( y_true= y_test, probas_pred= clf.predict_proba(X_test)[:, 0], pos_label=0)

    # ROC Curve

    fp_rate, tp_rate, threshold = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:, 0], pos_label=0)

    # Balanced accuracy
    balanced_acc = metrics.balanced_accuracy_score(y_test, predictions)

    # Regular accuracy
    acc = metrics.accuracy_score(y_test, predictions)

    # Precision

    precision = metrics.precision_score(y_true=y_test, y_pred=predictions, pos_label=0)

    recall = metrics.recall_score(y_true=y_test, y_pred=predictions)

    # Confusion matrix
    confusion_mat = metrics.confusion_matrix(y_test, clf.predict(X_test))

    # F-1 Score
    f_1 = metrics.f1_score(y_test, clf.predict(X_test))

    print(f"Accuracy: {acc}\n")
    print(f"Balanced Accuracy: {balanced_acc}\n")
    print(f"F-1 Score:{f_1}\n")
    print(f"Precision:{precision}\n")
    print(f"Recall:{recall}\n")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    ax1.matshow(confusion_mat, cmap=plt.cm.Blues, alpha=0.4)
    ax1.set_xlabel('Predictions')
    ax1.set_ylabel('Actual')
    ax1.set_title("Confusion Matrix")

    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax1.text(x=j, y=i, s=confusion_mat[i, j], va='center', ha='center', size='xx-large')

    ax2.plot(fp_rate, tp_rate)
    ax2.plot(np.array([0, 1]), color='orange')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title("ROC Curve")

    ax3.plot(recalls, precisions)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title("PR Curve")

    plt.tight_layout()

    if save_name == None:
        plt.show()
    else:
        fig.savefig(f'{save_name}.png')

    '''
    
    fig, axs = plt.subplots(3, figsize=(15, 15))

    axs[0].matshow(confusion_mat, cmap=plt.cm.Blues, alpha = 0.4)
    axs[0].set_title("Confusion Matrix")

    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            axs[0].text(x=j, y=i, s=confusion_mat[i, j], va='center', ha='center',size ='xx-large')

    axs[1].plot(precision, recall)
    axs[1].set_title("P-R Curve")

    axs[2].plot(fp_rate, tp_rate)
    axs[2].set_title("ROC Curve")

    plt.show()
    
    '''


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


def show_scree_plot(yes_imgs, no_imgs, max_components = 20):




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
    fig, axs = plt.subplots(2, 4)

    axs[0, 0].imshow(yes_before)
    axs[0, 0].set_title('Yes Before')
    axs[0, 1].imshow(yes_after)
    axs[0, 1].set_title('Yes After')
    axs[1, 0].imshow(no_before)
    axs[1, 0].set_title('No Before')
    axs[1, 1].imshow(no_after)
    axs[1, 1].set_title('No After')

    yes_before = yes_imgs[1].reshape(70, 70)
    pca = PCA(n_components=components)
    yes_x = pca.fit_transform(yes_imgs)
    yes_after = pca.inverse_transform(yes_x)[1].reshape(70, 70)

    no_before = no_imgs[1].reshape(70, 70)
    pca = PCA(n_components=components)
    no_x = pca.fit_transform(no_imgs)
    no_after = pca.inverse_transform(no_x)[1].reshape(70, 70)

    axs[0, 2].imshow(yes_before)
    axs[0, 2].set_title('Yes Before')
    axs[0, 3].imshow(yes_after)
    axs[0, 3].set_title('Yes After')
    axs[1, 2].imshow(no_before)
    axs[1, 2].set_title('No Before')
    axs[1, 3].imshow(no_after)
    axs[1, 3].set_title('No After')

    plt.show()

    return yes_x, no_x

def pca(yes_imgs, no_imgs, components = 20):

    pca = KernelPCA(n_components=components, kernel='rbf')
    yes_x = pca.fit_transform(yes_imgs)

    pca = KernelPCA(n_components=components, kernel='rbf')
    no_x = pca.fit_transform(no_imgs)

    return yes_x, no_x

def knn(X_train, y_train, X_test, y_test, save_name = None):

    params_dict = {'n_neighbors':np.arange(int(X_train.shape[0] / 10), int(X_train.shape[0] / 3)),
                   'p':[1, 2],
                   'metric': ['minkowski','l1', 'l2']
                   }

    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn,
                               params_dict,
                               verbose=1,
                               return_train_score=True)

    grid_search.fit(X_train, y_train)
    print(f"KNN Train Acc:{grid_search.score(X_train, y_train)}")
    print(f"KNN Test Acc:{grid_search.score(X_test, y_test)}")

    print("----------- Evaluation On Test Set ----------\n")

    score_model(grid_search, X_test, y_test, save_name)


def svm(X_train, y_train, X_test, y_test):

    default_gamma = 1 / (X_train.shape[1] * np.var(X_train))

    params_dict = {'kernel': ['poly', 'rbf', 'sigmoid'],
                   'C':[0.0001, 0.001, 0.01, 1, 10],
                   'gamma': [default_gamma * 0.1, default_gamma, default_gamma * 10]}

    svm = SVC(class_weight='balanced')

    grid_search = GridSearchCV(svm,
                               params_dict,
                               verbose=1,
                               return_train_score=True)

    grid_search.fit(X_train, y_train)


    print(f"SVM Acc:{grid_search.score(X_test, y_test)}")

def randomForest(X_train, y_train, X_test, y_test):

    depths = 20

    accs = []

    for d in range(depths):
        clf = RandomForestClassifier(max_depth= d + 1, random_state= 43, class_weight='balanced')
        clf.fit(X_train, y_train)
        accs.append(clf.score(X_test, y_test))

    plt.plot(np.arange(1, depths + 1), accs)
    plt.show()

    return accs
    #print(f"random forest acc:{clf.score(X_test, y_test)}")



def adaBoost(X_train, y_train, X_test, y_test):

    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, class_weight='balanced'))

    params_dict = {'learning_rate':[0.01, 0.05, 1, 5, 10]}

    grid_search = GridSearchCV(ada,
                               params_dict,
                               verbose=1,
                               return_train_score=True)

    grid_search.fit(X_train, y_train)
    print(f"Ada boost acc:{grid_search.score(X_test, y_test)}")


def get_train_test(yes_imgs, no_imgs):

    yes_label = np.zeros(len(yes_imgs))
    no_label = np.ones(len(no_imgs))

    final_label = np.concatenate((yes_label, no_label))
    final_imgs = np.concatenate((yes_imgs, no_imgs))

    assert(final_label.shape[0] == final_imgs.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(final_imgs, final_label, test_size = 0.15, random_state = 42)



    return X_train, y_train, X_test, y_test



if __name__ == '__main__':


    yes_path = 'C:\Dennis\Personal\Projects\ml_playground\data\\brain_mri\yes'
    no_path = 'C:\Dennis\Personal\Projects\ml_playground\data\\brain_mri\\no'

    yes_imgs = get_images(yes_path)
    no_imgs = get_images(no_path)

    yes_imgs = tools.flatten_imgs(yes_imgs)
    no_imgs = tools.flatten_imgs(no_imgs)

    print(f"After PCA (components = 6):")

    yes_imgs_pca, no_imgs_pca = pca(yes_imgs, no_imgs, 6)

    X_train, y_train, X_test, y_test = get_train_test(yes_imgs_pca, no_imgs_pca)

    knn(X_train, y_train, X_test, y_test, "6 components")

    print(f"After PCA (components = 5):")

    yes_imgs_pca, no_imgs_pca = pca(yes_imgs, no_imgs, 5)

    X_train, y_train, X_test, y_test = get_train_test(yes_imgs_pca, no_imgs_pca)

    knn(X_train, y_train, X_test, y_test, "5 components")

    print(f"After PCA (components = 4):")

    yes_imgs_pca, no_imgs_pca = pca(yes_imgs, no_imgs, 4)

    X_train, y_train, X_test, y_test = get_train_test(yes_imgs_pca, no_imgs_pca)

    knn(X_train, y_train, X_test, y_test, "4 components")

    print(f"After PCA (components = 3):")

    yes_imgs_pca, no_imgs_pca = pca(yes_imgs, no_imgs, 3)

    X_train, y_train, X_test, y_test = get_train_test(yes_imgs_pca, no_imgs_pca)

    knn(X_train, y_train, X_test, y_test, "3 components")

    print(f"Running PCA twice => 3 and then 4")

    yes_imgs_pca, no_imgs_pca = pca(yes_imgs, no_imgs, 3)

    yes_imgs_pca, no_imgs_pca = pca(yes_imgs_pca, no_imgs_pca, 4)

    X_train, y_train, X_test, y_test = get_train_test(yes_imgs_pca, no_imgs_pca)

    knn(X_train, y_train, X_test, y_test, "3 then 4 components")










