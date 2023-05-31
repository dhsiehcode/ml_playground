import numpy as np

import tools
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

p = 'C:\Dennis\Personal\Projects\ml_playground\data\credit_score\\application_record.csv'


def pca(p):
    df = tools.import_data(p)

    usable_cols = df.select_dtypes(include=int)

    # attempt PCA
    dim_nums = np.arange(len(usable_cols.columns))


    pca = PCA(n_components=len(dim_nums))
    pca.fit(usable_cols)

    plt.plot(dim_nums, pca.explained_variance_ratio_)
    plt.show()

from sklearn.cluster import KMeans












































