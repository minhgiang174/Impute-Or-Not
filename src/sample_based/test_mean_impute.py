import numpy as np
import sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss


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

        return np.argmin(error)

def cross_entropy_loss(y_true, y_pred):

    losses = []
    for i in range(len(y_true)):
        y = y_true[i]
        p = y_pred[i][1]
        losses.append(-(y * np.log(p) + (1 - y)*np.log(1 - p)))

    assert np.mean(losses) == log_loss(y_true, y_pred), np.mean(losses)
    return losses

if __name__ == "__main__":

    features = np.array([

    [1,2,3],
    [1,4,5],
    [2,3,7],
    [1,2,4]


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
    [4, np.nan, 8],
    [7, 8, 9],
    [np.nan, np.nan, 3]


    ])

    test_labels = np.array([

    [0],
    [0],
    [0],
    [1],
    [1]


    ])

    print(type(features))


    # qbc = QueryByCommittee()
    mean_selector = MeanImpute(None, loss = cross_entropy_loss)

    x = mean_selector.select(features, labels, missing_data, test_labels)
    print(x)
