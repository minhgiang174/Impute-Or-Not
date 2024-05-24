import numpy as np
import sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss

class SampleSubsetAcquisition():

    """
    Like mean imputation, but test all possible
    values for features (or a subset of them)

    """

    def __init__(self, max_subset_size, importance_selector,
    max, min, bins = 3, model = LogisticRegression, error = False):

        self.max_subset_size = max_subset_size
        self.importance_selector = importance_selector
        self.model = model
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.max = max
        self.min = min
        self.bins = bins
        self.error = error

        return

    def build_normal_bins(self, features, labels):

        """
        gets the mean of each feature. Assuming normal
        distribution of each column, make bins as
        mean +- std * i for all i in bin size

        if bin size is even, just remove mean

        """

        # Maybe sould do mean and stdv for each class?
        means = np.mean(features, axis = 0)
        stdvs = np.std(features, axis = 0)
        bins = [[means[j] + stdvs[j] * (i + 1) for i in range(-self.bins // 2, self.bins // 2)] for j in range(len(means))]
        return bins

    def repeat_and_fill(self, array, subset):

        array = np.reshape(array, (1, len(array)))
        array = np.tile(array, (len(subset), 1))

        print("array shape under rep fill\n", array.shape)
        print("subset shape\n", subset.shape)

        j = 0
        for col in range(len(array[0])):
            if np.isnan(array[0][col]):
                array[:, col] = subset[:, j]
                j += 1

        return array

    def select(self, features, labels, missing_data, test_labels):

        bins = np.array(self.build_normal_bins(features, labels))
        normalizer = self.scaler.fit(features)
        features_norm = normalizer.transform(features)
        model = self.model().fit(features_norm, labels.ravel())
        scores = [0 for i in range(len(missing_data))]

        for d in range(len(missing_data)):
            true_label = test_labels[d]
            possible_values = self.get_possible_values(missing_data[d], bins)
            possible_values_norm = normalizer.transform(possible_values)
            pred = model.predict_proba(possible_values_norm)


            print("pred\n", pred)

            print("score", self.importance_selector(missing_data[d], pred, true_label, model))

            scores[d] = self.importance_selector(missing_data[d], pred, true_label, model)

        print("scores\n", scores)
        if self.error:
            return np.argmin(scores) # actually error

        return np.argmax(scores)

    def get_possible_values(self, missing_data_point, bins):

        """
        TODO : Implement this

        Input: missing_data_point : a sample with some columns as np.nan
               bins : For each np.nan i in missing_data_point, the bins[i]
               has all the possible values it could be


        Output: all the possible permutations of missing_data_point with
                filled in values where the column is np.nan


        """

        print("bins ", bins)

        subset = self.yield_subset(bins, self.max_subset_size, missing_data_point)

        print("subset", subset)

        subset = self.repeat_and_fill(missing_data_point, subset)

        return subset

    def yield_subset(self, bins, size, missing_data_point):

        na_bins = np.where(np.isnan(missing_data_point))

        if len(bins[na_bins]) == 1:
            return np.reshape(bins[na_bins][0], (-1, 1))

        print("na_bins\n", na_bins)
        print("bins", bins.shape, "\n",  bins[na_bins])

        if len(bins) == 3:
            print(bins[0])
            print()
            print(bins[1])
            print()
            print(bins[2])
            print()

        print("missing_data_point\n", missing_data_point)

        iterator = itertools.product(*bins[na_bins])
        objs = np.zeros((min(size, len(bins[0]) ** len(bins[na_bins])), len(bins[na_bins])))
        print("objs shape", objs.shape)

        l = 0
        for p in iterator:
            print("p", p)
            objs[l] = p
            l += 1
            if l >= size:
                del iterator
                return objs

        del iterator
        return objs

def cross_entropy_selector(missing_data, pred, true_label, model):

    print("cross_entropy_selector called")
    print("true label\n", [true_label[0] for i in range(len(pred))])
    print("predicted probabilities\n", pred)

    return log_loss([true_label[0] for i in range(len(pred))], pred, labels = [0, 1])


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
    [np.nan, 8, 9],
    [np.nan, np.nan, 3]


    ])

    test_labels = np.array([

    [0],
    [0],
    [0],
    [1],
    [1]


    ])
    #
    # for i in range(-3//2 + 1, 3//2 + 1, 1):
    #     print(i)


    # subset_acquisitor = FeatureSubsetAcquisiiton(1, None, None, None, None)


    # subset_acquisitor.build_normal_bins(features, labels)

    # labels = np.array([1 for i in range(4)])
    # predicted = np.array([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.4, 0.6]])
    #
    # print(log_loss(labels, predicted, labels = [0, 1]))


    """
    Testing select method

    want nas to be filled in correctly.

    test acquisition function later...

    """


    sampl = SampleSubsetAcquisition(max_subset_size = 100,
                                    importance_selector =  cross_entropy_selector,
                                    max =  10,
                                    min =  0,
                                    bins =  3,
                                    model = LogisticRegression,
                                    error = True)


    sampl.select(features, labels, missing_data, test_labels)
