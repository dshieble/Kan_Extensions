import os
import gzip
import abc
from collections import namedtuple

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


       

class PreorderClassifier(object):
    """
    Classifier that applies a transform to data and then uses the left/right Kan classifiers
    """
    RAN = "ran"
    LAN = "lan"

    def __init__(self, *args, **kwargs):
        self._has_been_fit = False
        self.classes_ = [0,1]
        self.set_params(*args, **kwargs)

    def set_params(self, f: "Callable", get_f_learner: "Callable", verbose: bool, get_kind: "Callable"):
        assert get_f_learner is None or f is None
        self.f = f
        self.get_f_learner = get_f_learner
        self.verbose = verbose
        self.get_kind = get_kind

    def get_params(self, *args, **kwargs):
        return {
            "f": self.f,
            "get_f_learner": self.get_f_learner,
            "verbose": self.verbose,
            "get_kind": self.get_kind
        }
        
    def predict_row(self, row):
        if self.kind == PreorderClassifier.RAN:
            out = ~np.max(np.min(row <= self.X_false, axis=1))
        elif self.kind == PreorderClassifier.LAN:
            out = np.max(np.min(row >= self.X_true, axis=1))
        else:
            raise ValueError("{} not recognized".format(self.kind))
        return out

    def learn_f_learner(self, X, y):
        assert self.f is None
        assert self.get_f_learner is not None
        print("Fitting f_learner")
        f_learner = self.get_f_learner(X)
        f_learner.fit(X, y)
        self.f = lambda x: f_learner.predict(x)

    def transform_X(self, X):
        if self.f is not None:
            X = self.f(X)
        return X

    def fit(self, X, y):
        print("PreorderClassifier Fit is called with X.shape: {}".format(X.shape))
        self.kind = self.get_kind()
        self._has_been_fit = True
        if self.get_f_learner is not None:
            self.learn_f_learner(X, y)
        else:
            assert False
        X = self.transform_X(X)
        self.X_true = X[np.array(y, dtype=bool)]
        self.X_false = X[np.logical_not(np.array(y, dtype=bool))]
    
    def predict(self, X):
        assert self._has_been_fit
        X = self.transform_X(X)
        out = []
        iterator = tqdm(range(X.shape[0])) if self.verbose else range(X.shape[0])
        for i in iterator:
            out.append(self.predict_row(X[i, :]))
        return np.array(out).astype(np.float32)
    
    def predict_proba(self, X):
        predictions = self.predict(X)[:, None]
        return np.hstack([1-predictions, predictions])
        

class AbstractLearner(object):
    def __init__(self):
        __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, x, training):
        pass
    
class ClassifierProbabilityLearner(AbstractLearner):
    def __init__(self, clf_class, *args, **kwargs):
        self.clf = clf_class(*args, **kwargs)
        super().__init__()

    @abc.abstractmethod
    def fit(self, X, y, *args, **kwargs):
        self.clf.fit(X, y)

    @abc.abstractmethod
    def predict(self, X, *args, **kwargs):
        return self.clf.predict_proba(X)[:, 1, None]
#         return self.clf.predict(X)

class AbstractOrderingLossLearner(AbstractLearner):
    """
    Wrapper for learning a transformation via the ordering loss
    """
 
    def __init__(
        self, learning_rate, output_dimension=20,
        verbose=False, epochs=10, batches_per_epoch=10, *args, **kwargs):
        __metaclass__ = abc.ABCMeta
        import tensorflow as tf
        assert tf.executing_eagerly() # Required for learner construction
        self.output_dimension = output_dimension
        self.verbose = verbose
        self.batches_per_epoch = batches_per_epoch
        self.epochs = epochs
        self.model = self.get_model(*args, **kwargs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        super().__init__()

    @abc.abstractmethod
    def get_model(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_trainable_variables(self):
        pass
    
    @abc.abstractmethod
    def predict_tf(self, X, training):
        pass
    
    def predict(self, X, training):
        return self.predict_tf(X, training).numpy()
 
    def get_ordering_loss(self, X_true, X_false, training=True):
        return tf.reduce_sum(tf.math.maximum(0.0, 
                        tf.math.reduce_max(tf.transpose(self.predict_tf(X_false, training=training)), axis=1) -
                        tf.math.reduce_min(tf.transpose(self.predict_tf(X_true, training=training)), axis=1)))

    def train_step(self, X_true, X_false):
        with tf.GradientTape() as tape:
            loss_value = self.get_ordering_loss(X_true=X_true, X_false=X_false, training=True)
            grads = tape.gradient(loss_value, self.get_trainable_variables())
        self.optimizer.apply_gradients(zip(grads, self.get_trainable_variables()))
        return loss_value.numpy()

    def fit(self, X, y):
        X = X.astype(np.float64)
        y = y.astype(bool)

        X_true = X[np.array(y, dtype=bool)]
        X_false = X[np.logical_not(np.array(y, dtype=bool))]

        batch_size_true = X_true.shape[0] // self.batches_per_epoch
        batch_size_false = X_false.shape[0] // self.batches_per_epoch

        losses = []
        tracker = tqdm if self.verbose else lambda x: x
        for epoch in tracker(range(self.epochs)):
            for i in range(self.batches_per_epoch):
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
        
    def predict_tf(self, X, training):
        assert len(X.shape) == 2
        assert X.shape[1] == 784
        return self.model(np.reshape(X, [-1, 28, 28, 1]), training=training)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    
class NetworkOrderingLossLearner(AbstractOrderingLossLearner):
    
    def get_model(self, num_layers=1, hidden_layer_size=100):
        layers = [tf.keras.layers.Dense(hidden_layer_size) for i in range(num_layers)]
        return tf.keras.Sequential(layers + [tf.keras.layers.Dense(self.output_dimension)])
        
    def predict_tf(self, X, training=False):
        assert len(X.shape) == 2
        return self.model(X, training=training)

    def get_trainable_variables(self):
        return self.model.trainable_variables


class LinearOrderingLossLearner(AbstractOrderingLossLearner):

    def get_model(self, num_columns):
        return tf.Variable(2*(np.random.random((self.output_dimension, num_columns)) - 0.5), dtype=tf.float64)
        
    def predict_tf(self, X, **kwargs):
        assert len(X.shape) == 2
        assert X.shape[1] == self.model.shape[1]
        return tf.matmul(X, tf.transpose(self.model))

    def get_trainable_variables(self):
        return [self.model]
