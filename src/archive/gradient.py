import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def crazy_method(missing, base_learner, holdout, labels, loss_func):

    """
    Using a base learner trained on full data,
    suggest inputted values for spaces in missing
    that contain NA.

    Optimization: minimize expected risk under the base learner

    Metric: Holdout dataset

    """

    mask = generate_mask(missing)

    # Set nan features to zero (or something else)
    with_imputed_zero, where = impute_indices(mask, missing)

    model = base_learner().fit(with_imputed_zero, labels)

    gradient = 1
    while np.sum(abs(gradient)) > 0.1:

        gradient = risk_gradient(with_imputed_zero, loss_func, model, 0.1, where)

        gradient_mask = gradient * mask

        with_imputed_zero = update_params(with_imputed_zero, gradient_mask, lr = 0.1)

    return with_imputed_zero

def impute_indices(mask, missing):

    where = (np.where(mask))

    for i in range(len(where[0])):
        missing[where[0][i]][where[1][i]] = np.random.normal(np.nanmean(missing[:,where[1]]), 1) # can set to random too

    return missing, where

def learn(full_data, classes):

    """
    Given dataset with no missing values, train
    a base learner to minimize risk on.

    Input: full_data (np.darray), class of learner

    Output: learner of class "class" from scikit

    """

    model = None


    return model

def data_split(full_data, split):

    """
    separates full data into training and final holdout
    for validation. Split percentage by split.

    Input: full_data, split (float)

    Output: holdout data, validation set

    """

    test = None
    val = None

    return test, val


def generate_mask(imputed_data):

    indexes = np.isnan(imputed_data)

    return indexes




def compute_expected_risk(data, loss, model, where):

    """
    Given some data, compute the risk (test loss)
    with loss function loss_func.

    Input: data (numpy.ndarray), loss_func (R^(m x n) -> R^1), model

    Output: risk

    """

    classes = model.classes_

    risk = 0

    predicted_classes = model.predict_proba(data)

    for c in range(len(classes)):
        for k in range(len(classes)):
            classes_c = [classes[c] for i in range(len(classes))]
            risk += loss(y_true = classes_c, y_pred = predicted_classes, labels = classes)

    return risk


def risk_gradient(data, loss_func, model, step_size, where):

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
        gradient[where[0][i]][where[1][i]] = (compute_expected_risk(data + mask, loss_func, model, where) -
        compute_expected_risk(data, loss_func, model, where))/step_size
        mask[where[0][i]][where[1][i]] -= step_size

    # gradient = (compute_expected_risk(data + step_size, loss_func, model) -
    # compute_expected_risk(data, loss_func, model))/step_size

    # print("risks:")
    # print(compute_expected_risk(data, loss_func, model))
    # print(compute_expected_risk(data + step_size, loss_func, model))


    return gradient

def update_params(data, gradient, lr):


    """
    Update the parameters via gradient descent

    """

    # take a step against the gradient
    data -= lr * gradient


    return data


if __name__ == "__main__":
    #
    print("Testing functions")

    imputed_data = np.array([

    [1,2,3],
    [4,np.nan,6],
    [7,8,np.nan]

    ])

    og_data = np.array([

    [1,2,3],
    [4,5,6],
    [7,8,9]

    ])

    labels = np.array([1,2,0])

    # print(imputed_data)
    #
    #
    # mask = generate_mask(imputed_data)
    #
    # model = LogisticRegression().fit(mask, labels)
    # print(model.predict(mask))
    #
    # print(model.coef_)
    #
    #
    # model2param = LogisticRegression().fit(X = og_data, y = labels).coef_
    # print(model2param)


    res = crazy_method(imputed_data, LogisticRegression, None, labels, log_loss)

    print(res)
