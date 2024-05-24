import numpy as np
from tqdm import tqdm

def TrainRegressorForSingleFeature(data, feat_num, samples_to_impute, classifier):
    """
    Input:
    data (np.ndarray): The (feature) data. N x s. N = number of samples. s = number of features.
    feat_num (int): The feature to train the classifier on.
    samples_to_impute (list[int]): List of row indexes corresponding to samples that we should impute.
    classifier (sklearn estimator): The regressor/classifier used to impute values.

    Ouput:
    bool, Fitted sklearn classifier.
    """
    clf = classifier
    target=data[:, feat_num]
    features=np.delete(data, feat_num, axis=1)

    target = np.delete(target, samples_to_impute, axis = 0)
    features = np.delete(features, samples_to_impute, axis = 0)
    classes = np.unique(target)
    if len(classes) > 1:
        return True, clf.fit(features,target)
    else:
        return False, classes

def ImputeMissingValuesSingleFeature(data, feat_num, samples_to_impute, classifier):
    """
    Input:
    data (np.ndarray): The (feature) data. N x s. N = number of samples. s = number of features.
    feat_num (int): The feature to train the classifier on.
    samples_to_impute (list[int]): List of row indexes corresponding to samples that we should impute.
    classifier (sklearn estimator): The regressor/classifier used to impute values.

    Output:
    New data with imputed values.
    """

    use_clf, clf = TrainRegressorForSingleFeature(data, feat_num, samples_to_impute, classifier)
    if use_clf:
        feat_impute_samps = data[samples_to_impute]
        feat_impute_samps = np.delete(feat_impute_samps,feat_num,axis=1)
        imputed_feats = clf.predict(feat_impute_samps)
    else:
        # if use_clf is false, the second return argument is actually just the singular class value.
        imputed_feats = clf

    data[samples_to_impute, feat_num] = imputed_feats
    return data

def InitializeMissingValues(data, feat_num, samples_to_impute, seed):
    """
    Initializes missing values. For a given feature, the missing values will be drawn from a uniform distribution 
    ranging from the observed minimum and maximum of the feature.
    Input:
    data (np.ndarray): The (feature) data. N x s. N = number of samples. s = number of features.
    feat_num (int): The feature to train the classifier on.
    samples_to_impute (list[int]): List of row indexes corresponding to samples that we should impute.
    seed (int): The seed for the random number generator.

    Output:
    None
    """
    minimum, maximum = np.nanmin(data[:,feat_num]), np.nanmax(data[:,feat_num])
    rng = np.random.default_rng(seed)

    data[samples_to_impute, feat_num] = rng.uniform(minimum, maximum, len(samples_to_impute))

def ImputeDataMice(orig_data, classifier, threshold: float=1e-2, n_iters: int=10, seed: int=1337):
    """
    Input 
    orig_data (np.ndarray): The (feature) data. N x s. N = number of samples. s = number of features.
    classifier (sklearn estimator): The regressor/classifier used to impute values.
    threshold (float): The threshold for considering whether a value is converged. Default is 1e-2 (1% change).
        When no individual value changes more than the threshold, the data is considered converged. 
    n_iters (int): The maximum number of iterations to run mice for before stopping. Default is 10.
    seed (int): The seed for the random number generator.

    Output:
    New data with imputed values.
    """
    n_col = orig_data.shape[1]
    mask = np.isnan(orig_data)
    samples_to_impute = [np.where(mask[:, i]==True)[0] for i in range(n_col)]

    data = orig_data.copy()
    # Initialize missing values with sort of random values.
    for col in range(n_col):
        if len(samples_to_impute[col]) != 0:
            InitializeMissingValues(data, col, samples_to_impute[col], seed)
    
    cols_to_impute = np.array([col for col in range(n_col) if len(samples_to_impute[col])!= 0])

    # Prep variables for the while loop.
    previous_imputed_values = np.array([val for col in cols_to_impute for val in data[samples_to_impute[col], col].reshape(-1)])
    converged = False
    iter = 0

    while iter < n_iters and not converged:
        start_idx = 0 # used to keep track of current position in the vector of previous values.

        mask1 = np.ones(cols_to_impute.shape, dtype = bool) # Used to determine the columns for imputation for next cycle.
        mask2 = np.ones(previous_imputed_values.shape, dtype = bool) # Used to get the previous iterations imputed values.
        progress_bar = tqdm(cols_to_impute, desc=('Iteration %d' % iter), position=0, leave=False)
        for i, col in enumerate(progress_bar):
            if len(samples_to_impute[col]) != 0:
                ImputeMissingValuesSingleFeature(data, col, samples_to_impute[col], classifier)
                new_vals = data[samples_to_impute[col], col]
                old_vals = previous_imputed_values[start_idx: start_idx+len(samples_to_impute[col])] 
                
                # When all the new imputed values for a feature change by less than some threshold proportion of the
                # previous imputed values. Then we consider the feature converged.
                delta = np.abs(new_vals - old_vals) - threshold*old_vals 
                feature_converged = np.sum(delta > 0) == 0

                # If the feature is converged, remove this column from the list of columns that need to be imputed.
                if feature_converged:
                    mask1[i] = False
                    mask2[start_idx:start_idx+len(samples_to_impute[col])] = False
                    start_idx += len(samples_to_impute[col])
                    samples_to_impute[col] = np.array([])
                else:
                    previous_imputed_values[start_idx: start_idx+len(samples_to_impute[col])] = new_vals
                    start_idx += len(samples_to_impute[col])
            
        if start_idx != len(previous_imputed_values): # start index should always end up being the length of previous imputed values
            raise Exception('There is an bug in the code...')

        cols_to_impute = cols_to_impute[mask1] # Columns to impute for the next cycle.
        previous_imputed_values = previous_imputed_values[mask2]

        converged = len(cols_to_impute) == 0 # When no more columns to impute we end.
        iter += 1

    if converged: print("Converged and Finished!")
    else: print("Finished!")
    return data