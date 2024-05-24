import numpy as np
import basemodel

from tqdm import tqdm

class SEU(basemodel.Dataset):
    def __init__(self,
                x:np.ndarray=None, # N x d. N = number of samples. d = number of dimensions.
                y:np.ndarray=None, # N x c. N = number of samples. c = number of classes.
                seed:int=174, # The seed for generating the labeled and unlabeled sets
                train_test_idxs:tuple[np.ndarray, np.ndarray]=None, # Labeled set and unlabeled set indicies
                test_proportion:float=0.7, # Initial proportion of data to use as labeled set.                
                batch_size:int=1, # batch size.
                classifier:any=None, # Classifier used.
                *args,
                **kwargs,
                ) -> None:
        """
        Class for single feature active learning.
        Input:
        features (np.ndarray): N x d. N = number of samples. d = number of dimensions.
        labels (np.ndarray): N x c. N = number of samples. c = number of classes.
        seed (int): The seed for generating the labeled and unlabeled sets. Default is 0.
        ls_inds (tuple[np.ndarray, np.ndarray]): The row indicies corresponding to the training and testing data. Default is None.
            The first array is the indicies of the training (labeled) set.
            The second array is the indicies of the test (unlabeled) set.
        test_proportion (float): Initial proportion of data to use as testing (unlabeled) set. Default is 0.7.
        batch_size (int): Number of missing features to impute each iteration.
        sample_method (str): The sampling method. 'US' uniform sampling. 'ES' error sampling.
        classifier (sklearn classifier): Classifier used.
        """
        super().__init__(x, y, seed)

        self.x_cp = self.x.copy() # The copy of the data that will be modified.
        self.y_cp = self.y.copy() # The copy of the data labels that will be modified.

        if train_test_idxs is None:
            self.test_proportion = test_proportion
            self.train_idx, self.test_idx = self.split_data(self.test_proportion)
        else:
            assert len(train_test_idxs)==2, \
                "train_test_idxs must be a tuple of two np.ndarrays. The first array is the indicies of the training" \
                + "(labeled) set. The second array is the indicies of the test (unlabeled) set."
            self.test_proportion = len(train_test_idxs[1])/self.len() 
            self.train_idx, self.test_idx = train_test_idxs

        self.clf=classifier
        self.set_batchsize(batch_size)

    def set_batchsize(self, batch_size):
        self.batch_size = batch_size

    def getTrainData(self):
        return self.x_cp[self.train_idx], self.y_cp[self.train_idx]
    
    def getTestData(self):
        return self.x_cp[self.test_idx], self.y_cp[self.test_idx]

    def UpdateTrainSet(self, i, j, val):
        """
        Updates the value of the labeled set.
        """
        train_set = self.x_cp[self.train_idx]
        try:
            iter(val)
        except:
            train_set[i,j] = val
        else:
            for r, c, v in zip(i, j, val):
                train_set[r,c] = v
        self.x_cp[self.train_idx] = train_set

    def LogGainOneValue(self, i:int, j:int, value:float):
        """
        Calculates the log gain from replacing missing value at position (i,j) with value.
        Input:
        i (int): The row of the missing value
        j (int): The column of the missing value
        value (float): The value to try for the missing value at (i,j)

        Output:
        The log gain from replacing the missing feature with value.
        """
        train_set, train_set_labels = self.getTrainData()
        train_set_mod = train_set.copy() # labeled set copy
        train_set_mod[i,j] = value # labeled set copy with feature[i,j] set to value

        self.clf.fit(train_set_mod, train_set_labels)

        test_set, test_set_labels = self.getTestData()
        probs = self.clf.predict_proba(test_set)
        probs += 1e-10 # Small value to prevent log of 0.
        probs /= np.sum(probs, axis=1, keepdims=True)

        log_gain = np.sum(-np.log(probs[:,test_set_labels]))
        return log_gain
    
    def ChooseNextImputeValue(self, batch_size:int=None):
        raise NotImplementedError()
    
class SEU_US(SEU):
    def __init__(self, x: np.ndarray = None,
                y: np.ndarray = None,
                seed: int = 174,
                train_test_idxs: tuple[np.ndarray, np.ndarray] = None,
                test_proportion: float = 0.7,
                batch_size: int = 1,
                classifier: any = None,
                *args,
                **kwargs) -> None:
        super().__init__(x, y, seed, train_test_idxs, test_proportion, batch_size, classifier, *args, **kwargs)

    def ChooseNextImputeValue(self, batch_size:int=None):
        """
        Chooses the next missing value to replace
        Input:
        batch_size(int): The number of features to impute at once. 

        Output:
        (list[tuple[int, int, float]]): List of tuples. Each tuple is of the form (i, j, val). i and j are the row 
        and column of the missing value. Float is the best value to replace the missing value with.
        """
        if batch_size is not None: self.set_batchsize(batch_size)
        train_set = self.x_cp[self.train_idx]
        querries = np.nonzero(np.isnan(train_set)) # The position of the missing values. In the format of (row_inds, col_inds)
        querries = np.array([tup for tup in zip(*querries)])
        if len(querries) > 50:
            querries = self.rng.choice(querries, size=50)

        possible_vals_all_features = [np.unique(train_set[~np.isnan(train_set[:,i]),i]) for i in range(train_set.shape[1])] # Get the possible values for each feature
        scores = []
        for (i,j) in querries: # i'th sample, j'th feature
            possible_vals = possible_vals_all_features[j]
            prob = 1/len(possible_vals)
            score = prob*np.sum([self.LogGainOneValue(i,j,val) for val in possible_vals])
            scores.append(score)

        next_positions = querries[np.argsort(scores)[-self.batch_size:]]
        i = np.array([val[0] for val in next_positions])
        j = np.array([val[1] for val in next_positions])
        res = (i,j)

        return res
    
class SEU_ES(SEU):
    def __init__(self, x: np.ndarray = None,
                y: np.ndarray = None,
                seed: int = 174,
                train_test_idxs: tuple[np.ndarray, np.ndarray] = None,
                test_proportion: float = 0.7,
                batch_size: int = 1,
                classifier: any = None,
                *args,
                **kwargs) -> None:
        super().__init__(x, y, seed, train_test_idxs, test_proportion, batch_size, classifier, *args, **kwargs)

    def ChooseNextImputeValue(self, batch_size:int=None):
        """
        Chooses the next missing value to replace
        Input:
        batch_size(int): The number of features to impute at once. 

        Output:
        (list[tuple[int, int, float]]): List of tuples. Each tuple is of the form (i, j, val). i and j are the row 
        and column of the missing value. Float is the best value to replace the missing value with.
        """
        if batch_size is not None: self.set_batchsize(batch_size)
        train_set = self.x_cp[self.train_idx]
        querries = np.nonzero(np.isnan(train_set)) # The position of the missing values. In the format of (row_inds, col_inds)
        querries = np.array([tup for tup in zip(*querries)])
        if len(querries) > 50:
            x_train, y_train = self.getTrainData()
            self.clf.fit(x_train, y_train)
            probs = self.clf.predict_proba(x_train)
            r = np.repeat(np.arange(len(y_train)), 2)
            c = np.argsort(probs, axis=1)[:,-2:].flatten()
            weights = probs[r, c].reshape(-1, 2)
            weights = np.diff(weights).squeeze()
            weights = np.array([weights[tup[0]] for tup in querries])
            weights /= np.sum(weights)
            querries = self.rng.choice(querries, size=50, p=weights)

        possible_vals_all_features = [np.unique(train_set[~np.isnan(train_set[:,i]),i]) for i in range(train_set.shape[1])] # Get the possible values for each feature
        scores = []
        for (i,j) in querries: # i'th sample, j'th feature
            possible_vals = possible_vals_all_features[j]
            prob = 1/len(possible_vals)
            score = prob*np.sum([self.LogGainOneValue(i,j,val) for val in possible_vals])
            scores.append(score)

        next_positions = querries[np.argsort(scores)[-self.batch_size:]]
        i = np.array([val[0] for val in next_positions])
        j = np.array([val[1] for val in next_positions])
        res = (i,j)

        return res


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    import sklearn
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    # Load Data
    # Synthetic Data
    seed = 2024
    rng = np.random.default_rng(seed)

    n_samples, n_features = 100, 10
    data = rng.integers(0, 5, size=(n_samples, n_features))
    labels = rng.integers(0, 3,size=n_samples)
    print(data.shape, labels.shape)

    mask=np.zeros(n_samples*n_features, dtype=int)
    mask[:int(n_samples*n_features*0.5)]=1
    rng.shuffle(mask)
    mask = mask.astype(bool)
    mask = mask.reshape(n_samples, n_features)
    
    data = data.astype(float)
    data[mask] = np.nan
    
    # Test making SingleFeatureAL.
    classifier = RandomForestClassifier(n_estimators=50, max_depth = 3, random_state=147)
    model = SingleNaNAL(data, labels, classifier=classifier)
    trainData = model.getTrainData()
    res = model.ChooseNextImputeValue(1)
    model.UpdateTrainSet(*res)
    print(res)