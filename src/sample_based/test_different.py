import numpy as np
import sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss

def cross_entropy_selector(missing_data, pred, true_label, model):
    try:
        return log_loss([true_label[0] for i in range(len(pred))], pred, labels = [0, 1])
    except IndexError:
        return log_loss([true_label for i in range(len(pred))], pred, labels = [0, 1])

def cross_entropy_loss(y_true, y_pred):

    losses = []
    for i in range(len(y_true)):
        y = y_true[i]
        p = y_pred[i][1]
        losses.append(-(y * np.log(p) + (1 - y)*np.log(1 - p)))

    assert np.mean(losses) == log_loss(y_true, y_pred), np.mean(losses)
    return losses

class MeanImpute():

    """
    Imputes each cell with the known mean
    of the column, then selects via
    a query selection method.

    """

    def __init__(self, importance_selector, loss, model = LogisticRegression):

        self.importance_selector = importance_selector
        self.model = model
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.loss = loss
        return

    def select(self, features, labels, missing_data, test_labels):

        means = np.mean(features, axis = 0)
        imputed = np.where(np.isnan(missing_data), means, missing_data)
        imputed_norm = self.scaler.fit(imputed).transform(imputed)
        features_norm = self.scaler.fit(features).transform(features)
        model = self.model().fit(features_norm, labels.ravel())

        # Calculate expected model performance
        predicted_probs = model.predict_proba(imputed_norm)

        error = self.loss(test_labels, predicted_probs)

        print("scores sampler \n", error, "\n", np.argmin(error), "\n")

        return np.argmin(error)

class SamplingBasedSampleAcquisition():

    """
    Sample possible features from a Gaussian

    TODO : Implement discrete sampling

    """

    def __init__(self, sample_size, importance_selector,
    model = LogisticRegression, error = False, discrete_values = None,
    discrete_value_sets = None):

        self.sample_size = sample_size # N
        self.importance_selector = importance_selector
        self.model = model
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.error = error
        self.discrete_values = discrete_values
        self.discrete_value_sets = discrete_value_sets

        return

    def sample_N_times(self, array, means, stdvs):

        if self.discrete_values is None:

            return self.sample_only_normal(array, means, stdvs)

        array = np.reshape(array, (1, len(array)))
        array = np.tile(array, (self.sample_size, 1))


        for col in range(len(array[0])):

            if np.isnan(array[0][col]):

                if self.discrete_values[col]:
                    column = np.random.choice(self.discrete_value_sets[col],
                    size = self.sample_size, replace = True)

                else:
                    column = np.random.normal(means[col], stdvs[col], size = self.sample_size)

                array[:, col] = column

        return array

    def sample_only_normal(self, array, means, stdvs):

        array = np.reshape(array, (1, len(array)))
        array = np.tile(array, (self.sample_size, 1))

        for col in range(len(array[0])):

            if np.isnan(array[0][col]):

                array[:, col] = np.random.normal(means[col], stdvs[col], size = self.sample_size)

        return array

    def select(self, features, labels, missing_data, test_labels):

        # bins = np.array(self.build_normal_bins(features, labels))
        means = np.mean(features, axis = 0)
        stdvs = np.std(features, axis = 0)
        normalizer = self.scaler.fit(features)
        features_norm = normalizer.transform(features)
        model = self.model().fit(features_norm, labels.ravel())
        scores = [0 for i in range(len(missing_data))]

        for d in range(len(missing_data)):
            true_label = test_labels[d]
            possible_values = self.sample_N_times(missing_data[d], means, stdvs)
            possible_values_norm = normalizer.transform(possible_values)
            pred = model.predict_proba(possible_values_norm)
            scores[d] = self.importance_selector(missing_data[d], pred, true_label, model)

        print("scores sampler \n", scores, "\n", np.argmin(scores), "\n")
        if self.error:
            return np.argmin(scores) # actually error

        return np.argmax(scores)

if __name__ == "__main__":
    print()

    features = np.array([

    [1,2,0],
    [1,4,1],
    [2,3,0],
    [1,2,1]


    ])

    labels = np.array([

    [0],
    [1],
    [1],
    [0]

    ])

    missing_data = np.array([

    [np.nan, 1, 1],
    [2, 3, np.nan],
    [4, np.nan, 0],
    [np.nan, 8, 1],
    [np.nan, np.nan, 1],
    [1, np.nan, np.nan]


    ])

    test_labels = np.array([

    [0],
    [0],
    [0],
    [1],
    [1],
    [0]


    ])

    discrete_values = [False, False, True]
    discrete_value_sets = [[], [], [0, 1]]

    sampler = SamplingBasedSampleAcquisition(sample_size = 10,
                                                   importance_selector = cross_entropy_selector,
                                                   discrete_values = discrete_values,
                                                   discrete_value_sets = discrete_value_sets)


    mean_selector = MeanImpute(None, loss = cross_entropy_loss)


    sampler.select(features, labels, missing_data, test_labels)

    mean_selector.select(features, labels, missing_data, test_labels)
