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
from umap import UMAP

UMAP_ALGO = "umap"
TSNE_ALGO = "tsne"
FASHION_PATH = "/Users/dshiebler/workspace/personal/Category_Theory/Kan_Extensions/data/fashion"

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



def get_2d_mnist_dataset(num_train_images, num_test_images, transform, pca_components=100):
    X_mnist_train_raw, y_mnist_train, X_mnist_test_raw, y_mnist_test = get_mnist_dataset(
        num_train_images=num_train_images, num_test_images=num_test_images)
    y_train, y_test = np.int64(y_mnist_train), np.int64(y_mnist_test)

    if transform == TSNE_ALGO:
        steps = [("PCA", PCA(n_components=pca_components)), ("TSNE", TSNE(n_components=2, n_jobs=-1))]
        Xtransformed = Pipeline(steps).fit_transform(np.vstack((X_mnist_train_raw, y_train)))
        X_train, X_test = Xtransformed[:X_mnist_train_raw.shape[0]], Xtransformed[X_mnist_test_raw.shape[0]:]
    elif transform == UMAP_ALGO:
        steps = [("PCA", PCA(n_components=pca_components)), ("UMAP", UMAP(n_components=2, n_jobs=-1))]
        pipeline = Pipeline(steps).fit(X_mnist_train_raw, y_train)
        X_train = pipeline.transform(X_mnist_train_raw)
        X_test = pipeline.transform(X_mnist_test_raw)
    else:
        raise ValueError("transform {} not recognized".format(transform))
    return X_train, X_test, y_train, y_test


def split_dataset(X, y, tr_proportion):
    num_train_samples = int(X.shape[0] * tr_proportion)
    indices = np.random.permutation(np.arange(0, X.shape[0]))
    tr_indices, te_indices = indices[:num_train_samples], indices[num_train_samples:]
    return X[tr_indices], y[tr_indices], X[te_indices], y[te_indices]

def get_mag_dataset(tr_proportion=0.8, samples=100):
    X = np.random.random((samples, 3))
    mins = np.min(X, axis=1)
    y = mins > np.median(mins)
    return split_dataset(X, y, tr_proportion)
    
def get_wine_dataset(tr_proportion=0.8, target_label=None):
    wine_dataset = load_wine()
    X, y = wine_dataset["data"], wine_dataset["target"]
    if target_label is not None:
        y = (y == target_label)
    return split_dataset(X, y, tr_proportion)
