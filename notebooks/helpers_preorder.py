import os
import gzip
import abc
from collections import namedtuple

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score


Results = namedtuple("Results", "train_tpr train_tnr test_tpr test_tnr")


def true_positive_rate(truth, preds):
    # Returns the true positive rate, aka recall or sensitivity
    tp = sum([1 if preds[i] and truth[i] else 0 for i in range(len(truth))])
    fn = sum([1 if not preds[i] and truth[i] else 0 for i in range(len(truth))])
    return tp / (tp + fn)


def true_negative_rate(truth, preds):
    # Returns the true negative rate, aka specificity
    tn = sum([1 if not preds[i] and not truth[i] else 0 for i in range(len(truth))])
    fp = sum([1 if preds[i] and not truth[i] else 0 for i in range(len(truth))])
    return tn / (tn + fp)


def evaluate(classifier_class, X_train, y_train, X_test, y_test, f):
    """
    Evaluate a preorder classifier by fitting it to testing data and computing the true positive and
        true negative rates on training and testing data
    Args:
        classifier_class: Class that inherits from AbstractPreorderClassifier
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        f: Function to use to transform the data
    Returns
        Classification results
    """
    clf = classifier_class(f=f)
    clf.fit(X_train, y_train)
    return Results(
        train_tpr=true_positive_rate(y_train, clf.predict(X_train)),
        train_tnr=true_negative_rate(y_train, clf.predict(X_train)),
        test_tpr=true_positive_rate(y_test, clf.predict(X_test)),
        test_tnr=true_negative_rate(y_test, clf.predict(X_test)),
    )
       

class AbstractPreorderClassifier(object):
    """
    Classifier that applies a transform to data and then uses the left/right Kan classifiers
    """
    
    def __init__(self, verbose=False, f=None):
        self.verbose = verbose
        self.f = f
    
    def transform_X(self, X):
        if self.f is not None:
            X = self.f(X)
        return X

    def fit(self, X, y):
        X = self.transform_X(X)
        self.X_true = X[np.array(y, dtype=bool)]
        self.X_false = X[np.logical_not(np.array(y, dtype=bool))]
    
    def predict_row(self, row):
        raise NotImplementedError

    def predict(self, X):
        X = self.transform_X(X)
        out = []
        tracker = tqdm if self.verbose else lambda x: x
        for i in tqdm(range(X.shape[0])):
            out.append(self.predict_row(X[i, :]))
        return np.array(out)
        

class RanPreorderClassifier(AbstractPreorderClassifier):
    kind = "ran"

    def predict_row(self, row):
        return ~np.max(np.min(row <= self.X_false, axis=1))


class LanPreorderClassifier(AbstractPreorderClassifier):
    kind = "lan"

    def predict_row(self, row):
        return np.max(np.min(row >= self.X_true, axis=1))
    

class AbstractOrderingLossLearner(object):
    """
    Wrapper for learning a transformation via the ordering loss
    """
 
    def __init__(self, learning_rate, *args, output_dimension=20, verbose=False, **kwargs):
        __metaclass__ = abc.ABCMeta
        assert tf.executing_eagerly() # Required for learner construction
        self.output_dimension = output_dimension
        self.verbose = verbose
        self.model = self.get_model(*args, **kwargs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    @abc.abstractmethod
    def get_model(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_trainable_variables(self):
        pass
    
    @abc.abstractmethod
    def predict(self, x, training):
        pass
    
    def get_ordering_loss(self, X_true, X_false, training=True):
        return tf.reduce_sum(tf.math.maximum(0.0, 
                        tf.math.reduce_max(tf.transpose(self.predict(X_false, training=training)), axis=1) -
                        tf.math.reduce_min(tf.transpose(self.predict(X_true, training=training)), axis=1)))

    def train_step(self, X_true, X_false):
        with tf.GradientTape() as tape:
            loss_value = self.get_ordering_loss(X_true=X_true, X_false=X_false, training=True)
            grads = tape.gradient(loss_value, self.get_trainable_variables())
        self.optimizer.apply_gradients(zip(grads, self.get_trainable_variables()))
        return loss_value.numpy()

    def fit(self, X, y, epochs=10, batches_per_epoch=10):
        X = X.astype(np.float64)
        y = y.astype(bool)

        X_true = X[np.array(y, dtype=bool)]
        X_false = X[np.logical_not(np.array(y, dtype=bool))]

        batch_size_true = X_true.shape[0] // batches_per_epoch
        batch_size_false = X_false.shape[0] // batches_per_epoch

        losses = []
        tracker = tqdm if self.verbose else lambda x: x
        for epoch in tracker(range(epochs)):
            for i in range(batches_per_epoch):
                loss = self.train_step(
                    X_true=X_true[i:i+batch_size_true],
                    X_false=X_false[i:i+batch_size_false])
                losses.append(loss)
        return losses

        
class ConvOrderingLossLearner(AbstractOrderingLossLearner):
    
    def get_model(self):
        return tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, [3,3], activation='relu',
                                 input_shape=(None, None, 1)),
          tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
          tf.keras.layers.GlobalAveragePooling2D(),
          tf.keras.layers.Dense(self.output_dimension)
        ])
        
    def predict(self, X, training):
        assert len(X.shape) == 2
        assert X.shape[1] == 784
        return self.model(np.reshape(X, [-1, 28, 28, 1]), training=training)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    
class NetworkOrderingLossLearner(AbstractOrderingLossLearner):
    
    def get_model(self, num_layers=1, hidden_layer_size=100):
        layers = [tf.keras.layers.Dense(hidden_layer_size) for i in range(num_layers)]
        return tf.keras.Sequential(layers + [tf.keras.layers.Dense(self.output_dimension)])
        
    def predict(self, X, training):
        assert len(X.shape) == 2
        return self.model(X, training=training)

    def get_trainable_variables(self):
        return self.model.trainable_variables


class LinearOrderingLossLearner(AbstractOrderingLossLearner):

    def get_model(self, num_columns):
        return tf.Variable(2*(np.random.random((self.output_dimension, num_columns)) - 0.5), dtype=tf.float64)
        
    def predict(self, X, **kwargs):
        assert len(X.shape) == 2
        assert X.shape[1] == self.model.shape[1]
        return tf.matmul(X, tf.transpose(self.model))

    def get_trainable_variables(self):
        return [self.model]
