import os
import gzip
import abc


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


FASHION_PATH = "../data/fashion"
TRAIN_SIZE = 1000
TEST_SIZE = 100

def load_mnist(path, kind='train'):


    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
        
    images = StandardScaler().fit_transform(images)
    return images, labels


def get_mnist_dataset(included_classes=None, num_train_images=-1, num_test_images=-1):
    raw_Xtrain, raw_ytrain = load_mnist(FASHION_PATH, kind='train')
    raw_Xtest, raw_ytest = load_mnist(FASHION_PATH, kind='t10k')

    included_classes = included_classes if included_classes is not None else set(raw_ytrain)
    
    train_indices = np.random.permutation(
        [i for i in np.arange(len(raw_ytrain)) if raw_ytrain[i] in included_classes])[:num_train_images]
    raw_Xtrain = raw_Xtrain[train_indices]
    raw_ytrain = raw_ytrain[train_indices]
    
    test_indices = np.random.permutation(
        [i for i in np.arange(len(raw_ytest)) if raw_ytest[i] in included_classes])[:num_test_images]
    raw_Xtest = raw_Xtest[test_indices]
    raw_ytest = raw_ytest[test_indices]
    rawX = np.vstack((raw_Xtrain, raw_Xtest))
    return raw_Xtrain, raw_ytrain, raw_Xtest, raw_ytest


class TransformedSampleProvider(object):

    def __init__(self):
        from umap import UMAP

        self.X_train_all_raw, self.y_train_all_raw, self.X_test_all_raw, self.y_test_all_raw = get_mnist_dataset(
            num_train_images=TRAIN_SIZE * 10,
            num_test_images=TEST_SIZE * 10)
        self.y_train_all, self.y_test_all = np.int64(self.y_train_all_raw), np.int64(self.y_test_all_raw)
        self.pipeline = Pipeline(
            [("PCA", PCA(n_components=100)), ("UMAP", UMAP(n_components=2, n_jobs=-1))])
    
    def get_transformed_sample(self):
        indices_tr = np.random.permutation(range(self.X_train_all_raw.shape[0]))[:TRAIN_SIZE]
        indices_te = np.random.permutation(range(self.X_test_all_raw.shape[0]))[:TEST_SIZE]

        self.pipeline.fit(self.X_train_all_raw[indices_tr], self.y_train_all[indices_tr])
        X_train = self.pipeline.transform(self.X_train_all_raw[indices_tr])
        X_test = self.pipeline.transform(self.X_test_all_raw[indices_te])
        return X_train, self.y_train_all[indices_tr], X_test, self.y_test_all[indices_te]
    