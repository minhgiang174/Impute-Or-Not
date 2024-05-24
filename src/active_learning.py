import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import mice
from sklearn.linear_model import LogisticRegression

class MultipleImputater():
    def __init__(self, m) -> None:
        """
        m: number of imputations.
        """
        self.m = m
        self.x_raw = None# Masked data (n, f)
        self.x_intermediate = None# Imputed data (m, n, f)
        self.x_imputed = None# Imputed data (n, f)

    def impute(self, x):
        """
        x (np.ndarray): n x f matrix. n = number of samples. f = number of features.
        """
        self.x_raw = x
        result = []
        for i in range(self.m):
            result.append(mice.ImputeDataMice(self.x_raw, classifier=LogisticRegression(max_iter=1000), seed=i*174))
        self.x_intermediate = np.array(result)
        self.x_imputed = np.mean(self.x_intermediate, axis=0)

        return self.x_imputed    

def Implement_ActiveLearningWithImputationWithUncertainty(X, Y, train_idx, test_idx, imputation_method, imputation_m, seeds, batch_k=1, query_method='random', end_portion=0.5):
    '''
    Implement over different seeds
    '''
    all_unseen_accs = []
    for seed in seeds:
      unseen_acc = ActiveLearningWithImputationWithUncertainty(X, Y, train_idx, test_idx, imputation_method, imputation_m, seed, batch_k, query_method, end_portion)
      all_unseen_accs.append(unseen_acc)

    return np.array(all_unseen_accs)

def ActiveLearningWithImputationWithUncertainty(X, Y, train_idx, test_idx, imputation_method, imputation_m, seed, batch_k=1, query_method='random', end_portion=0.5):
    '''
    This function implement Imputation With Uncertainty from paper from Jongmin Han, Seokho Kang
    All of the X can have nan data.
    Args:
    - X
    - Y
    - imputation_method: currently we have only mice
    - imputation_m: number of simulation for imputation
    - seed
    - batch_k: 1 by default
    - query_method='random'
    - end_portion=0.5
    Returns:
    - unseen_accuracies: the accuracy of model on unseen data over time until the stopping condition.
    '''

    # Set seed
    np.random.seed(seed)
    n = len(X)

    train_idxs = train_idx.copy()
    remain_idxs = test_idx.copy()

    # Multiple Imputation Phase
    ## In this phase, use MICE to generate m and take average of that m values for each missing data
    ## The imputater here should fit on the train data and then return all unlabel data with imputated values
    imputater = MultipleImputater(imputation_m)
    X_imputated = imputater.impute(X) # Use full data to impute the missing values

    # tqdm
    batch_bar = tqdm(total=int(end_portion*n)-len(remain_idxs), dynamic_ncols=True, leave=True, position=0, desc=f'AL seed {seed}')

    # Active Learning Phase
    ## This phase, pick a batch of k from the imputated X and use the query method to pick the next instance
    unseen_accuracies = []

    while len(remain_idxs) > int(end_portion*n): # run to half of the data
        X_train, y_train = X_imputated[train_idxs], Y[train_idxs]
        X_remain, y_remain = X_imputated[remain_idxs], Y[remain_idxs]

        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=147)
        model.fit(X_train, y_train)

        preds = model.predict(X_remain)
        unseen_acc = accuracy_score(preds, y_remain)
        unseen_accuracies.append(unseen_acc)

        next_idx = SelectNextIndices(model, X_remain, y_remain, query_method, batch_k, imputater)
        actual_next_idx = remain_idxs[next_idx] # a list because we select a batch of k

        # Now update train and remain indices
        train_idxs = np.concatenate((train_idxs, actual_next_idx))
        remain_idxs = np.setdiff1d(remain_idxs, actual_next_idx)

        batch_bar.set_postfix(acc="{:.04f}".format(float(unseen_acc)))
        batch_bar.update()

    return np.array(unseen_accuracies)

def SelectNextIndices(model, X, y, query_method, k, MultipleImputater):
    '''
    Select the k next indice(s) using given query_method
    Args:
    - model: classification model
    - X: data
    - y: label
    - query_method: all possible methods - random, entropy, least_confidence, margin
    - k: batch size
    Returns:
    - A list of k indices that selected, corresponding to given X and y.
    '''
    n = np.arange(len(X))
    if query_method == 'random':
        next_idxs = np.random.choice(n, size=k)

    elif query_method == 'entropy':
        next_idxs = Query_Selection_Entropy(model, X, k)

    elif query_method == 'least_confidence':
        next_idxs = Query_Selection_Confidence(model, X, k)

    elif query_method == 'margin':
        next_idxs = Query_Selection_Margin(model, X, k)

    elif query_method == 'gini':
        next_idxs = Query_Selection_GINI(model, X, k)

    elif query_method == 'cumir':
        # assert MultipleImputater != None
        next_idxs = Query_Selection_CUMIR(model, X, k, MultipleImputater)

    return next_idxs

def Entropy(class_prob):
    '''
    Calculate entropy given all class predicted probabilities
    '''
    entropy = 0
    for prob in class_prob:
      if prob > 0:
        entropy += prob * np.log2(prob)
    return np.array(-entropy)

def Query_Selection_Entropy(model, X, k):
    '''
    Entropy query selection with a batch of with the highest entropy
    '''
    # 1. predict the probabilities of all classes
    pred_probs = model.predict_proba(X)
    num_sam = pred_probs.shape[0]

    # 2. Calculate entropies
    all_entropies = [Entropy(pred_probs[i]) for i in range(num_sam)]
    sorted_entropies = np.argsort(all_entropies)[::-1]

    return sorted_entropies[:k]

def Query_Selection_Confidence(model, X, k):
    '''
    Pick a batch of k sample with least confidence
    '''
    # 1. predict the probabilities of all classes
    pred_probs = model.predict_proba(X)
    num_sam = pred_probs.shape[0]
    # 2. Get the confidence probabilities for all samples among all classes
    neg_confidence = [-np.max(pred_probs[i,:]) for i in range(num_sam)]
    # 3. Get the most k confident
    sorted_confidence = np.argsort(neg_confidence)[::-1] # pick min of the max

    return sorted_confidence[:k]

def Query_Selection_Margin(model, X, k):
    '''
    Implement query selection with margin:
    pick the instance that have the least margin of confidence between the two most confident labels
    '''

    # 1. predict the probabilities of all classes
    pred_probs = model.predict_proba(X)
    num_sam, num_class = pred_probs.shape

    # 2. Get the least margin between the two most confident labels for each instance
    all_margins = [np.diff(np.sort(pred_probs[i,:])[-2:])[0] for i in range(num_sam)]

    # 3. Get least margin of all
    sorted_margins = np.argsort(all_margins)

    return sorted_margins[:k]

def Query_Selection_GINI(model, X, k):
    '''
    USe Gini to select the next instance
    '''
    # 1. predict the probabilities of all classes
    pred_probs = model.predict_proba(X)
    num_sam, num_class = pred_probs.shape

    # 2. get gini scores
    gini_uncertainty_scores = [1 - np.sum(np.power(pred_probs[i,], 2)) for i in range(num_sam)]

    # 3. Get k most uncertain
    sorted_gini = np.argsort(gini_uncertainty_scores)[::-1]

    return sorted_gini[:k]

def Query_Selection_CUMIR(model, X, k, Multiply_Imputater):
    '''
    USe CUMIR to select the next instance
    This method use additional multiply imputated values when we imputate the data in the beginning to calculate the imputation uncertainty
    '''
    m = Multiply_Imputater.m

    # Predict the probabilities of all classes for the final imputated X
    pred_probs = model.predict_proba(X) # (num_sample x num_classes)
    num_sam, num_class = pred_probs.shape

    # Use model and predict on all the intermediate imputated data
    m_intermediate_Xs = Multiply_Imputater.x_intermediate
    m_intermediate_pred_probs = np.array([model.predict_proba(m_intermediate_Xs[i]) for i in range(m)])

    # Get cumir score for each of the sample
    cumir_scores = [Calculate_CUMIR_single(pred_probs[i], m_intermediate_pred_probs[:,i,:], num_class, m) for i in range(num_sam)]

    # Get the index of sample with the highest cumir score (least uncertain)
    sorted_cumir_scores = np.argsort(cumir_scores)[::-1]

    # Return k
    return sorted_cumir_scores[:k]


def Calculate_CUMIR_single(xhat_preb_probs, m_x_pred_probs, num_classes, num_imputation):

    def ScoreFunction(all_probs, c):
        numerator = np.max(all_probs) - 1/c
        denom = 1 - 1/c
        scr = 1 - (numerator/denom)
        return scr

    def IndicatorFunc(xhat_probs, all_imputated_instance_probs):
        '''
        Indicator Function: checking if the argmax of final imputated values == the argmax of the sum of all m imputated data
        '''
        yhat = np.argsort(xhat_probs)[-1]
        sum_all_probs = np.sum(all_imputated_instance_probs, axis=0)
        ybar = np.argsort(sum_all_probs)[-1]
        return 1 if yhat == ybar else 0

    # Size check
    assert m_x_pred_probs.shape[0] == num_imputation
    assert m_x_pred_probs.shape[-1] == num_classes

    c = num_classes

    single_imputation_scores = [ScoreFunction(m_x_pred_probs[i], c) for i in range(num_imputation)]
    cumir_score = np.mean(single_imputation_scores)
    rubins_term = (1+1/num_imputation) * (1/(num_imputation-1)) * IndicatorFunc(xhat_preb_probs, m_x_pred_probs)

    return cumir_score + rubins_term
