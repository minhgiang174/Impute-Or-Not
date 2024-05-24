import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

class ExpectedRiskImputation():

    def __init__(self, labels, model = LogisticRegression, loss_func = log_loss):
        self.labels = labels
        self.base_learner = model
        self.loss_func = loss_func

    def __call__(self, missing, seed):

        mask = self.generate_mask(missing)

        # Set nan features to zero (or something else)
        with_imputed_zero, where = self.impute_indices(mask, missing)

        self.model = self.base_learner().fit(with_imputed_zero, self.labels)

        gradient = 1
        while np.sum(abs(gradient)) > 0.1:

            gradient = self.risk_gradient(with_imputed_zero, self.loss_func, 0.1, where)

            gradient_mask = gradient * mask

            with_imputed_zero = self.update_params(with_imputed_zero, gradient_mask, lr = 0.1)

        return with_imputed_zero

    def impute_indices(self, mask, missing):

        where = (np.where(mask))

        for i in range(len(where[0])):
            missing[where[0][i]][where[1][i]] = np.random.normal(np.nanmean(missing[:,where[1]]), 1) # can set to random too

        return missing, where

    def generate_mask(self, imputed_data):

        mask = np.isnan(imputed_data)

        return mask

    def compute_expected_risk(self, data, loss, where):

        """
        Given some data, compute the risk (test loss)
        with loss function loss_func.

        Input: data (numpy.ndarray), loss_func (R^(m x n) -> R^1), model

        Output: risk

        """

        classes = self.model.classes_

        classes_c = np.zeros(len(data))

        risk = 0

        predicted_classes = self.model.predict_proba(data)

        for c in range(len(classes)):
            for k in range(len(classes)):
                classes_c = classes_c * 0 + classes[c]
                risk += loss(y_true = classes_c, y_pred = predicted_classes, labels = classes)

        return risk

    def risk_gradient(self, data, loss_func, step_size, where):

        """
        Given data, compute the gradient of the loss by finite
        distance approximation.

        Input: data, step_size (mask of gradients to consider)
        loss_func and model to be input into compute_expected_risk

        Output: gradient vector for params

        """

        mask = np.zeros(data.shape)
        gradient = np.zeros(data.shape)

        for i in range(len(where[0])):

            mask[where[0][i]][where[1][i]] += step_size
            gradient[where[0][i]][where[1][i]] = (self.compute_expected_risk(data + mask, loss_func, where) -
            self.compute_expected_risk(data, loss_func, where))/step_size
            mask[where[0][i]][where[1][i]] -= step_size

        return gradient

    def update_params(self, data, gradient, lr):


        """
        Update the parameters via gradient descent

        """

        # take a step against the gradient
        data -= lr * gradient


        return data
