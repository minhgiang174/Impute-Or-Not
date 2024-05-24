import numpy as np
import sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class RandomSelection():

    """
    Query selection method.

    Just selects a random query.

    """

    def __init__(self):
        return

    def select(self, features, labels, missing_data, test_labels):

        return np.random.randint(len(missing_data))

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

        missing_data = np.array(missing_data)

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

    assert np.mean(losses) == log_loss(y_true, y_pred, labels = [0, 1]), np.mean(losses)
    return losses

class SampleSubsetAcquisition():

    """
    Like mean imputation, but test all possible
    values for features (or a subset of them)

    """

    def __init__(self, max_subset_size, importance_selector,
    max_ = 1, bins = 3, model = LogisticRegression, error = False,
    all_labels = [0, 1]):

        self.max_subset_size = max_subset_size
        self.importance_selector = importance_selector
        self.model = model
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.max_ = max_
        self.bins = bins
        self.error = error
        self.all_labels = all_labels

        return

    # def build_normal_bins(self, features, labels):
    #
    #     """
    #     gets the mean of each feature. Assuming normal
    #     distribution of each column, make bins as
    #     mean +- std * i for all i in bin size
    #
    #     if bin size is even, just remove mean
    #
    #     """
    #
    #     # Maybe sould do mean and stdv for each class?
    #     means = np.mean(features, axis = 0)
    #     stdvs = np.std(features, axis = 0)
    #     bins = [[means[j] + stdvs[j] * (i + 1) for i in range(-self.bins // 2, self.bins // 2)] for j in range(len(means))]
    #     return bins


    def build_normal_bins(self, features, labels):

        """
        gets the mean of each feature. Assuming normal
        distribution of each column, make bins as
        mean +- std * i for all i in bin size

        if bin size is even, just remove mean

        """

        if self.discrete_values is None:
            return self.build_only_numerical(features, labels)

        step = self.max_ /  ((self.bins - 1) // 2)
        # num_bins = 3
        # all_labels = [1,2,3]
        bins = {label : [] for label in self.all_labels}

        means = {}
        stdvs = {}

        for label in self.all_labels:

            means[label] = np.mean(features[np.where(labels == label)[0]], axis = 0)
            stdvs[label] = np.std(features[np.where(labels == label)[0]], axis = 0)

        for label in self.all_labels:

            for col in range(len(features[0])):

                if self.discrete_values[col]:
                    column = self.discrete_value_sets[col]

                else:
                    if stdvs[label][col] == 0:
                        column = [means[label][col]]
                    else:
                        column = [means[label][col] + stdvs[label][col] * (i + step) for i in range(-self.max_ -1, -self.max_ + self.bins - 1))]

                bins[label].append(column)


        return bins

    def build_only_numerical(self, features, labels):

        bins = {label : [] for label in self.all_labels}

        step = self.max_ /  ((self.bins - 1) // 2)

        means = {}
        stdvs = {}

        for label in all_labels:

            means[label] = np.mean(features[np.where(labels == label)[0]], axis = 0)
            stdvs[label] = np.std(features[np.where(labels == label)[0]], axis = 0)

        for label in all_labels:

            for col in range(len(features[0])):

                if stdvs[label][col] == 0:
                    column = [means[label][col]]
                else:
                    column = [means[label][col] + stdvs[label][col] * (i + step) for i in range(-self.max_ -1, -self.max_ + self.bins - 1))]

                bins[label].append(column)


        return bins

    def repeat_and_fill(self, array, subset):

        array = np.reshape(array, (1, len(array)))
        array = np.tile(array, (len(subset), 1))

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


            # print("pred\n", pred)
            #
            # print("score", self.importance_selector(missing_data[d], pred, true_label, model))

            scores[d] = self.importance_selector(missing_data[d], pred, true_label, model)

        # print("scores\n", scores)
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

        # print("bins ", bins)

        subset = self.yield_subset(bins, self.max_subset_size, missing_data_point)

        # print("subset", subset)

        subset = self.repeat_and_fill(missing_data_point, subset)

        return subset

    def yield_subset(self, bins, size, missing_data_point):

        na_bins = np.where(np.isnan(missing_data_point))

        if len(bins[na_bins]) == 1:
            return np.reshape(bins[na_bins][0], (-1, 1))

        # print("na_bins\n", na_bins)
        # print("bins", bins.shape, "\n",  bins[na_bins])

        # print("missing_data_point\n", missing_data_point)

        iterator = itertools.product(*bins[na_bins])
        objs = np.zeros((min(size, len(bins[0]) ** len(bins[na_bins])), len(bins[na_bins])))
        # print("objs shape", objs.shape)

        l = 0
        for p in iterator:
            # print("p", p)
            objs[l] = p
            l += 1
            if l >= size:
                del iterator
                return objs

        del iterator
        return objs

def cross_entropy_selector(missing_data, pred, true_label, model):

    try:
        return log_loss([true_label[0] for i in range(len(pred))], pred, labels = [0, 1])
    except IndexError:
        return log_loss([true_label for i in range(len(pred))], pred, labels = [0, 1])



class ImputationSimulation():

    """
    class to run the sample imputation simulation.

    Needs a query selection class (see RandomSelection for an example)

    requires initial features and labels (clean)

    split is another method (should make RandomSplit default) that
    ensures the initial set has at least one of each label. It also
    can start with an arbitrary percent of "clean" data.

    model is an sklearn class. For example, sklearn.linear_model.LogisticRegression
    or RandomForest etc. don't call the model when initializing ImputationSimulation.

    methods is a list of query selection methods. Each method needs a select
    attribute that takes in the missing data, their corresponding labels, and
    the clean data (features and labels). The initialization for the
    methods should be done prior to initializing ImputationSimulation. See
    __main__ for an example using RandomSelection. Also, the select method
    should return an integer, representing the index in the missing data to choose.

    TODO: Implement actual active learning algorithms. As long as it is a
    class with a select method, it should work.

    TODO: Make run_simulation work for batch selection as well. But this might
    be less of a priority (I should have code for the homework to do this).

    """

    def __init__(self, features, labels, split, percent_missing, model,
    methods, evaluation_mode = "all", fold = None):
        self.features = features
        self.labels = labels
        self.percent_missing = percent_missing
        self.model = model
        self.methods = methods
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.split = split
        self.evaluation_mode = evaluation_mode

        if self.evaluation_mode == "cross validation":
            if fold is None:
                self.fold = 3
            else:
                self.fold = fold

        return

    def add_and_remove(self, feat, label, feat_full, label_full,
    unlabel, test_labels, x, indexes):

        """
        Helper function to deal with adding samples
        from the unlabeled set to the labeled set.

        L: the set of intances with labels
        U: the set of instances without labels

        Input: feat: the current set of features in L
               label: the current set of labels in L
               feat_full: all the available features
               label_full: every label (these are unknown in practice)
               unlabel: the current set of features in U (missing features)
               x: the index to remove from unlabel
               indexes: index list such that index[x] is the index of the
                        point in the feat_full and label_full objects

        Output: feat with a single sample added
                label with a single sample added
                unlabel with a single sample removed
                indexes with a single index removed

        """

        feat_shape = feat.shape[1]

        unlabel = np.delete(unlabel, x, axis = 0)
        test_labels = np.delete(test_labels, x, axis = 0)

        label_to_get = np.array(indexes)[x]
        indexes = np.delete(indexes, x)

        label_to_add = label_full[label_to_get]
        feature_to_add = feat_full[label_to_get]

        if len(label_to_add.shape) == 0:
            label_to_add = np.reshape(label_to_add, (1, 1))[0]
            feature_to_add = np.reshape(feature_to_add, (1, feat_shape))
            label = np.append(label, label_to_add, axis = 0)
            feat = np.append(feat, feature_to_add, axis = 0)
        else:
            label = np.append(label, label_to_add)
            feat = np.vstack((feat, feature_to_add))

        return feat, label, unlabel, test_labels, indexes

    def randomly_na(self, data, percent_missing):

        """
        For each cell in the data, choose if
        it should be nan with some
        probability given by percent_missing.

        Require that percent_missing is in the domain [0, 1].
        if percent_missing is 0.2, for example, then there's a
        20% chance a given cell will be turned to np.nan

        """

        if percent_missing > 1:
            percent_missing = percent_missing / 100

        mask = np.random.rand(data.shape[0], data.shape[1])
        mask = mask < percent_missing
        data = (np.where(mask, np.nan, data))

        return data

    def evaluate(self, features_imputed, labels_imputed, model, features_norm, means, stdvs):

        """

        "all" method uses the current model to
        get the accuracy for the entire dataset

        "cross validation" does the cross val
        on only the current clean data (doesn't need
        to "cheat" by knowing the features of the missing
        data pool)

        """

        if self.evaluation_mode == "all":
            accuracy = self.accuracy_on_all_samples(self.features,
             self.labels, features_imputed, labels_imputed, model)
            means.append(accuracy)

        if self.evaluation_mode == "cross validation":
            scores = self.cross_validation(features_norm, labels_imputed.ravel(), model)

            if np.isnan(np.mean(scores)):
                scores = self.accuracy_on_all_samples(self.features,
                 self.labels, features_imputed, labels_imputed, model)

            means.append(np.mean(scores))
            stdvs.append(np.std(scores))


        return means, stdvs




    def run_simulation(self):

        """
        Ranges through each query selection method
        and returns the means and stdvs of each

        all simulations should start with the same data
        and use the same evaluation metric

        """

        # Make initial set with no missing data
        n = len(self.features)
        initial_set_indexes, missing_data_indexes = self.split(self.features, self.labels)

        # Initial_set_indexes : has all the indexes for the set with no missing data
        # missing_data_indexes : initial indexes for the data with missing values.
                                #also the indexes for the corresponding labels

        # split the data into the initial set and missing data set
        features_imputed = self.features[initial_set_indexes]
        labels_imputed = self.labels[initial_set_indexes]

        # create the missing data set. Should start the same for all simulations
        missing_features_all = self.randomly_na(self.features, self.percent_missing)

        missing_features = missing_features_all[missing_data_indexes]
        test_labels = self.labels[missing_data_indexes]

        # mean and standard deviation of accuracy for each selection method
        # in self.methods. note that stdvs will only be used when cross validation
        # evaluation method is used.
        all_means = []
        all_stdvs = []

        # Run simulation with the same initial set for many
        # possible imputation query selection methods
        for selection_method in self.methods: # each selection_method is an object


            # indexes corresponding to data currently with missing values
            # maps each index in missing_features to the full dataset (self.features)
            indexes = missing_data_indexes.copy()
            missing_features = missing_features_all[missing_data_indexes]
            test_labels = self.labels[missing_data_indexes]
            features_imputed = self.features[initial_set_indexes]
            labels_imputed = self.labels[initial_set_indexes]

            # print("RUNNING SIMULATION")
            # print(selection_method)
            # print("AYO! Shape check!")
            # print("indexes", len(indexes))
            # print("self.features", self.features.shape)
            # print("self.labels", self.labels.shape)
            # print("missing_features", missing_features.shape)
            # print("labels_imputed", labels_imputed.shape)
            # print("features_imputed", features_imputed.shape)

            means = []
            stdvs = []

            while len(labels_imputed) < n: # Stop at n, since there will be no disagreement on the last sample

                # mean center data everytime we fir a model
                features_norm = self.scaler.fit(features_imputed).transform(features_imputed)
                # Train model on current set of imputed (clean) data
                model = self.model().fit(features_norm, labels_imputed.ravel())

                # Get accuracy metrics for current clean data
                means, stdvs = self.evaluate(features_imputed, labels_imputed, model, features_norm, means, stdvs)

                # Call the oracle to select the next features to impute
                x = selection_method.select(features_imputed, labels_imputed,
                 missing_features, test_labels)

                # Add x to labeled_set, remove from unlabeled
                features_imputed, labels_imputed, missing_features, test_labels, indexes = self.add_and_remove(features_imputed,
                                                                                   labels_imputed,
                                                                                   self.features,
                                                                                   self.labels,
                                                                                   missing_features,
                                                                                   test_labels,
                                                                                   x,
                                                                                   indexes)

            # Add current run accracies to set of all simulations
            all_means.append(means)
            all_stdvs.append(stdvs)

        return all_means, all_stdvs

    def cross_validation(self, features, labels, model):

        """
        Split the dataset into train and test, compute the self.fold times cross validation.

        """

        scores = cross_val_score(model, features, labels, cv = self.fold)

        return scores

    def calc_loss(self, features_imputed_norm, labels_imputed, model):
        model_ = model.fit(features_imputed_norm, labels_imputed)
        pred = model_.predict(features_imputed_norm)
        return np.sum(pred == labels_imputed) / len(pred)


    def accuracy_on_all_samples(self, missing_features_imputed,
    test_labels, known_features, train_labels, model):
        known_features_norm = self.scaler.fit(known_features).transform(known_features)
        missing_features_imputed_norm = self.scaler.fit(missing_features_imputed).transform(missing_features_imputed)
        pred = model.predict(missing_features_imputed_norm)

        return np.sum(pred == test_labels) / len(pred)


class RandomSplit():

    """
    class that does stuff
    """

    def __init__(self, split_percent):
        self.split = split_percent

    def __call__(self, features, labels):
        all_ind = [i for i in range(len(features))]
        n = len(all_ind)

        indexes_samples_labeled = np.random.choice(all_ind, size = n, replace = False)

        # Keep track of indexes in unlabeled data so we add the correct points
        label_set_indexes = indexes_samples_labeled[0 : int(self.split * n)]
        indexes = indexes_samples_labeled[int(self.split * n) : n]

        for label in np.unique(labels):
            if np.sum(labels[label_set_indexes] == label) == 0:
                return self.__call__(features, labels)

        return label_set_indexes, indexes

def Implement_Random_Masking(data, sample_proportions, feature_proportion, seed):
    '''
    This function takes X and the masking portion and return the new X with the same shape with missing the given portion of random data
    Args:
    - data: full data
    - sample_proportions: float
    - feature_proportion: float
    - seed: int
    Returns:
    - masked_data: same shape as given data, missing given portion of data
    '''

    np.random.seed(seed)
    cols = np.random.choice(np.arange(data.shape[1]), size=int(feature_proportion*data.shape[1]))

    mask_indices = np.random.rand(*data[:, cols].shape) < sample_proportions
    masked_data = data.copy()
    masked_data_cols = masked_data[:, cols]
    masked_data_cols[mask_indices] = np.nan
    masked_data[:, cols] = masked_data_cols

    return masked_data

def str_to_int_labels(labels):

    strs = np.unique(labels)

    for i in range(len(strs)):
        labels = np.where(labels == strs[i], i, labels)

    return labels.astype(np.int32)

if __name__ == "__main__":
    data = pd.read_csv("fertility_Diagnosis.txt")

    matrix = np.array(data)
    labels = matrix[:, -1]
    features = matrix[:, 0:9]

    labels = str_to_int_labels(labels)

    split = RandomSplit(split_percent = 0.1)

    percent_missing = 0.1

    model = LogisticRegression

    methods = [RandomSelection()]

    impute = ImputationSimulation(features, labels, split, percent_missing, model, methods, "cross validation")

    predicted = LogisticRegression().fit(features, labels).predict(features)
    print(np.mean(predicted == labels))

    means, stds = impute.run_simulation()

    print(len(matrix))
    print(means)
