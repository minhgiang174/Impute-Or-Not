import numpy as np
import basemodel

class RFVA(basemodel.Dataset):
    def __init__(self,
                x:np.ndarray=None, # N x d. N = number of samples. d = number of dimensions.
                y:np.ndarray=None, # N x c. N = number of samples. c = number of classes.
                seed:int=174, # The seed for generating the labeled and unlabeled sets
                train_test_idxs:tuple[np.ndarray, np.ndarray]=None, # Labeled set and unlabeled set indicies
                test_proportion:float=0.7, # Initial proportion of data to use as labeled set.                
                batch_size:int=1, # batch size.
                classifier:any=None, # Classifier used.
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
        next_positions = self.rng.choice(querries, size=self.batch_size)
        i = np.array([val[0] for val in next_positions])
        j = np.array([val[1] for val in next_positions])
        res = (i,j)

        return res
