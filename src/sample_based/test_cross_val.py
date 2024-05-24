
import sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def FiveFoldCrossValidation(features, labels, model, fold = 3):

    """
    Split the dataset into train and test, compute the 5 fold cross validation.

    """

    scores = cross_val_score(model, features, labels, cv = fold)

    return scores

if __name__ == "__main__":

    data = pd.read_csv("fertility_Diagnosis.txt")
    matrix = np.array(data)
    labels = matrix[:, -1]
    features = matrix[:, 0:9]

    model = LogisticRegression().fit(features, labels.ravel())

    print(model.coef_)
    print()

    FiveFoldCrossValidation(features, labels.ravel(), model)

    print(model.coef_)
