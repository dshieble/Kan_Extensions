from copy import deepcopy
import tensorflow as tf
import numpy as np
import random
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.colors as mcolors

import pandas as pd
from functools import reduce
from collections import defaultdict
from scipy.spatial import distance_matrix


def add_to_transitive_closure(relation, new_group):
    """
    Given a partition of a space and a new set of points to be grouped together, return out_relation such that any
        pair of points that are together in relation or new_group are together in out_relation
    Args:
        relation (dict<int, set<index>>): A indexed collection of sets that covers some X
        new_group (set): A subset of X
    Returns:
        A list of sets that covers X
    """
    intersect_key = None
    intersects = [new_group]
    does_not_intersect = {}
    for key, existing_group in relation.items():
        if len(existing_group.intersection(new_group)) > 0:
            intersect_key = key if intersect_key is None else max(key, intersect_key)
            intersects.append(existing_group)
        else:
            does_not_intersect[key] = existing_group

    assert intersect_key is not None
    does_not_intersect[intersect_key] = reduce(lambda a, b: set.union(a, b), intersects)
    return does_not_intersect

def generate_predictions_from_relation(X_test, relation):
    """
    Args:
        X_test (np.array<float>): A numpy array of size (num_test_points, n_dim)
        relation (dict<int, set<int>>): An indexed partition of X_test
    Returns:
        A list<int> whose ith element is the index of the partition in relation that index i falls 
            into and is -1 if index i is not in relation
    """
    cluster_predictions = []
    for ix in range(X_test.shape[0]):
        ix_prediction = -1
        for key, cluster in relation.items():
            # Any point in a cluster with only one point should get marked as -1 
            if len(cluster) > 1 and ix in cluster:
                ix_prediction = key
        cluster_predictions.append(ix_prediction)
    return cluster_predictions


####### Incoming #########

def sample_incoming_nonexpansive_map_from_two_points_same_cluster(X_train, X_test, num_maps_per_cluster, y_train):
    """
    Sample incoming maps from two-point subsets of X_train into X_test
    Args:
        X_train (np.array<float>): A numpy array of size (num_train_points, n_dim)
        X_test (np.array<float>): A numpy array of size (num_test_points, n_dim)
        num_maps_per_cluster (int): Number of pairs to sample for each unique cluster in y_train
        y_train (np.array<int>): A numpy array of size (num_train_points, 1) 
    Returns:
        A list of pairs of indices [((tr1, tr2), (te1, te2))] such that
            distance(X_test[te1],X_test[te2]) <= distance(X_train[te1],X_train[te2])
    """
    D_train, D_test = distance_matrix(X_train, X_train), distance_matrix(X_test, X_test)

    # For each cluster, select num_maps_per_cluster pairs of points
    test_pairs = {(ix1, ix2) for ix1 in range(X_test.shape[0]) for ix2 in range(X_test.shape[0]) if ix1 != ix2}
    maps = []
    for cluster in tqdm(set(y_train)):
        cluster_indices = [ix for ix, c in enumerate(y_train) if c == cluster]
        train_cluster_pairs = [(ix1, ix2)
                               for ix1 in cluster_indices
                               for ix2 in cluster_indices if D_train[ix1][ix2] > 0]
        
        # Sampled pairs of training points in the same cluster, inversely chosen by distance
        weights = [np.exp(-D_train[ix1][ix2]) for (ix1, ix2) in train_cluster_pairs]
        sampled_train_pair_indices = np.random.choice(
            range(len(train_cluster_pairs)),
            size=num_maps_per_cluster,
            p=weights / np.sum(weights),
            replace=False)
        sampled_train_pairs = [train_cluster_pairs[ix] for ix in sampled_train_pair_indices]

        # For each sampled training pair, randomly select a testing pair that has lower distance
        for tr1, tr2 in sampled_train_pairs:
            eligible_test_pairs = [(ix1, ix2) for (ix1, ix2) in test_pairs
                    if D_test[ix1][ix2] <= D_train[tr1][tr2]]
            if len(eligible_test_pairs) > 0:
                (te1, te2) = random.choice(eligible_test_pairs)
                maps.append(((tr1, tr2), (te1, te2)))
                test_pairs.remove((te1, te2))
    return maps



def construct_relation_from_incoming_nonexpansive_maps(num_test_points, incoming_nonexpansive_maps, y_train):
    """
    Args:
        incoming_nonexpansive_maps (list<(int, int), (int, int)>): A list of pairs of indices [((tr1, tr2), (te1, te2))]
        y_train (np.array<int>): A numpy array of size (num_train_points, 1) 
    Returns:
        A dict<int, set<int>> representing an indexed partition of range(num_test_points) such that the
            points (te1, te2) are in the same partition whenever there exists some ((tr1, tr2), (te1, te2)) in 
            incoming_nonexpansive_maps
    """
    relation = {ix: {ix} for ix in range(num_test_points)}
    for m in incoming_nonexpansive_maps:
        cluster_to_indices = defaultdict(set)
        for ix in range(len(m[0])):
            cluster_of_train_point = y_train[m[0][ix]]
            index_of_test_point = m[1][ix]
            cluster_to_indices[cluster_of_train_point].add(index_of_test_point)
        for v in cluster_to_indices.values():
            relation = add_to_transitive_closure(relation, set(v))
    return relation











####### Outgoing #########

def get_relation_index_to_distance(D_test, relation):
    """   
    Args:
        X_test (np.array<float>): A numpy array of size (num_test_points, n_dim)
        relation (list<set>): A list of sets of indices in range(num_test_points)
    Returns:
        A len(relation) x len(relation) matric of floats who i,jth element is the maximum distance from a point
             in the ith partition of relation to a point in the jth partition
    """
    out = {}
    for k1 in relation.keys():
        for k2 in relation.keys():
            out[(k1, k2)] = np.max(D_test[list(relation[k1]), :][:, list(relation[k2])])
    return out


def sample_outgoing_nonexpansive_map_to_two_points_different_clusters(
        X_train, X_test, relation, num_maps_per_cluster_pair, y_train, num_cluster_pairs=None):
    """
    Sample outgoing maps from X_test into two-point subsets of X_train
    Args:
        X_train (np.array<float>): A numpy array of size (num_train_points, n_dim)
        X_test (np.array<float>): A numpy array of size (num_test_points, n_dim)
        relation (list<set>): A list of sets of indices in range(num_test_points)
        num_maps_per_cluster_pair (int): Number of maps to sample for each pair of clusters
        y_train (np.array<int>): A numpy array of size (num_train_points, 1) 
        num_cluster_pairs (int): The number of cluster pairs to sample. Sample all pairs if None
    Returns:
        A list [(list<int>, list<int>)] such that in each tuple (l1, l2) the elements of l1 are indices into relation
            and the elements of l2 are indices into relation representing the maps that sends all points in the
            partitions indexed by l1 to one point and all points in the partitions indexed by l2 to another point
    """
    D_train = distance_matrix(X_train, X_train)
    
    # relation_index_to_distance[(k1, k2)] is the maximum distance from a point in the key=k1 partition of relation to a 
    #      point in the key=k2 partition
    relation_index_to_distance = get_relation_index_to_distance(distance_matrix(X_test, X_test), relation)
    relation_index_pairs = {(k1, k2) for k1 in relation.keys() for k2 in relation.keys() if k1 != k2}

    # For each cluster, select num_maps_per_cluster pairs of points
    maps = []
    all_cluster_pairs = [(c1, c2) for c1 in set(y_train) for c2 in set(y_train) if c1 != c2]
    sampled_cluster_pairs = all_cluster_pairs if num_cluster_pairs is None else [all_cluster_pairs[ix]
                                 for ix in np.random.permutation(range(len(all_cluster_pairs)))[:num_cluster_pairs]]
    for cluster_1, cluster_2 in tqdm(sampled_cluster_pairs):
        cluster_1_indices = [ix for ix, c in enumerate(y_train) if c == cluster_1]
        cluster_2_indices = [ix for ix, c in enumerate(y_train) if c == cluster_2]
        train_cross_cluster_pairs = [(ix1, ix2)
                               for ix1 in cluster_1_indices
                               for ix2 in cluster_2_indices if D_train[ix1][ix2] > 0]

        # Sampled pairs of training points in the same cluster chosen proportionately by distance
        weights = [D_train[ix1][ix2] for (ix1, ix2) in train_cross_cluster_pairs]
        sampled_train_pair_indices = np.random.choice(
            range(len(train_cross_cluster_pairs)),
            size=num_maps_per_cluster_pair,
            p=weights / np.sum(weights),
            replace=False)
        sampled_train_pairs = [train_cross_cluster_pairs[ix] for ix in sampled_train_pair_indices]

        # For each sampled training pair, assign each partition in relation to one of the points in the pair
        for tr1, tr2 in sampled_train_pairs:
            eligible_relation_pairs = [(k1, k2) for (k1, k2) in relation_index_pairs
                    if relation_index_to_distance[(k1, k2)] >= D_train[tr1][tr2]]
            if len(eligible_relation_pairs) > 0:
                (r1, r2) = random.choice(eligible_relation_pairs)
                maps.append(((r1, r2), (tr1, tr2)))
                relation_index_pairs.remove((r1, r2))
    return maps


def construct_relation_from_outgoing_nonexpansive_maps(incoming_relation, outgoing_nonexpansive_maps, y_train):    
    """
    Args:
        incoming_relation (list<set>): A list of sets that covers some X
        outgoing_nonexpansive_maps (list<(int, int), (int, int)>): A list of pairs of indices [((p1, p2), (te1, te2))]
            where (p1,p2) are indices into incoming_relation
        y_train (np.array<int>): A numpy array of size (num_train_points, 1) 
    Returns:
        A consolidation of incoming_relation where any sets brided by outgoing_nonexpansive_maps are connected
    """
    relation = deepcopy(incoming_relation)
    for m in outgoing_nonexpansive_maps:
        relation = add_to_transitive_closure(
            relation=relation, new_group=set.union(incoming_relation[m[0][0]], incoming_relation[m[0][1]]))
    return relation
