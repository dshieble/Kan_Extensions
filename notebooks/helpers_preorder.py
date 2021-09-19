import os
import gzip
import abc

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score


def true_positive_rate(truth, preds):
    # Returns the true positive rate, aka recall
    tp = sum([1 if preds[i] and truth[i] else 0 for i in range(len(truth))])
    fn = sum([1 if not preds[i] and truth[i] else 0 for i in range(len(truth))])
    return tp / (tp + fn)

def true_negative_rate(truth, preds):
    # Returns the true negative rate
    tn = sum([1 if not preds[i] and not truth[i] else 0 for i in range(len(truth))])
    fp = sum([1 if preds[i] and not truth[i] else 0 for i in range(len(truth))])
    return tn / (tn + fp)

def true_positive_score(y_train, y_test):
    return recall_score(y_train, y_test)

def evaluate(X_train, y_train, X_test, y_test, f):
    metrics = {
        "roc_auc_score": roc_auc_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate
    }
    for kind in [RawPreorderClassifier.ran, RawPreorderClassifier.lan]:
        clf = RawPreorderClassifier(kind=kind, f=f)
        clf.fit(X_train, y_train)
        print("{} tr: {}".format(kind, {m: f(y_train, clf.predict(X_train)) for m, f in metrics.items()}))
        print("{} te: {}".format(kind, {m: f(y_test, clf.predict(X_test)) for m, f in metrics.items()}))
        print()

class RawPreorderClassifier(object):
    
    ran = "ran"
    lan = "lan"
    
    def __init__(self, kind, f=None):
        assert kind == self.ran or kind == self.lan
        self.kind = kind
        self.f = f
    
    def transform_X(self, X):
        if self.f is not None:
            X = self.f(X)
        return X

    def fit(self, X, y):
        X = self.transform_X(X)
        self.X_true = X[np.array(y, dtype=bool)]
        self.X_false = X[np.logical_not(np.array(y, dtype=bool))]
        
    def predict(self, X):
        X = self.transform_X(X)
        out = []
        for i in tqdm(range(X.shape[0])):
            if self.kind == self.ran:
                # RanK
                out.append(~np.max(np.min(X[i, :] <= self.X_false, axis=1)))
            else:
                # LanK
                out.append(np.max(np.min(X[i, :] >= self.X_true, axis=1)))
        return np.array(out)
    
    
    
 
###########################################################################


class AbstractLearner(object):

    def __init__(self, learning_rate, *args, **kwargs):
        __metaclass__ = abc.ABCMeta
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
    
    def get_loss(self, X_true, X_false, training=True):
        return tf.reduce_sum(tf.math.maximum(0, 
                        tf.math.reduce_max(tf.transpose(self.predict(X_false, training=training)), axis=1) -
                        tf.math.reduce_min(tf.transpose(self.predict(X_true, training=training)), axis=1)))

    def train_step(self, X_true, X_false):
        with tf.GradientTape() as tape:
            loss_value = self.get_loss(X_true=X_true, X_false=X_false, training=True)

        grads = tape.gradient(loss_value, self.get_trainable_variables())
        self.optimizer.apply_gradients(zip(grads, self.get_trainable_variables()))
        return loss_value.numpy()

    def fit(self, X, y, epochs=10, batches_per_epoch=10):

        X_true = X[np.array(y, dtype=bool)]
        X_false = X[np.logical_not(np.array(y, dtype=bool))]

        batch_size_true = X_true.shape[0] // batches_per_epoch
        batch_size_false = X_false.shape[0] // batches_per_epoch

        losses = []
        for epoch in tqdm(range(epochs)):
            for i in range(batches_per_epoch):
                loss = self.train_step(
                    X_true=X_true[i:i+batch_size_true],
                    X_false=X_false[i:i+batch_size_false])
                losses.append(loss)
        return losses

        
class ConvLearner(AbstractLearner):
    
    def get_model(self, m=10):
        return tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, [3,3], activation='relu',
                                 input_shape=(None, None, 1)),
          tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
          tf.keras.layers.GlobalAveragePooling2D(),
          tf.keras.layers.Dense(m)
        ])
        
    def predict(self, X, training):
        assert len(X.shape) == 2
        assert X.shape[1] == 784
        return self.model(np.reshape(X, [-1, 28, 28, 1]), training=training)

    def get_trainable_variables(self):
        return self.model.trainable_variables

class NetworkLearner(AbstractLearner):
    
    def get_model(self, num_layers=1, hidden_layer_size=100, m=10):
        layers = [tf.keras.layers.Dense(hidden_layer_size) for i in range(num_layers)]
        return tf.keras.Sequential(layers + [tf.keras.layers.Dense(m)])
        
    def predict(self, X, training):
        assert len(X.shape) == 2
        return self.model(X, training=training)

    def get_trainable_variables(self):
        return self.model.trainable_variables

class LinearLearner(AbstractLearner):

    def get_model(self, num_columns, m=10):
        return tf.Variable(2*(np.random.random((m, num_columns)) - 0.5))
        
    def predict(self, X, **kwargs):
        assert len(X.shape) == 2
        assert X.shape[1] == self.model.shape[1]
        return tf.matmul(X, tf.transpose(self.model))

    def get_trainable_variables(self):
        return [self.model]