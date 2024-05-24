from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import os
from sklearn.metrics import mean_squared_error as mse

class Dataset():
  def __init__(self, x, y, seed:int=174):
    self.x = x
    self.y = y
    self.set_seed(seed)

  def len(self):
    return len(self.x)
  
  def __len__(self):
    return len(self.x)

  def set_seed(self, seed:int=174):
    self.rng = np.random.default_rng(seed)

  def split_data(self, test_portion, stratify: bool=False, seed:int=None):
    if seed is not None: self.set_seed(seed)

    n = self.x.shape[0] # Number of samples
    if stratify:
      classes = np.unique(self.y)
      group_idxs = [np.argwhere(self.y==label).flatten() for label in classes]
      test_idx = [self.rng.choice(idxs, size=int(test_portion*len(idxs)), replace=False) for idxs in group_idxs]
      test_idx = np.sort(np.concatenate(test_idx))
      train_idx = np.setdiff1d(np.arange(n), test_idx)
    else:
      test_idx = self.rng.choice(np.arange(n), size = int(test_portion*n), replace=False)
      train_idx = np.setdiff1d(np.arange(n), test_idx)
    return train_idx, test_idx

  def get_split(self, test_portion, seed:int=None):
    train_idx, test_idx = self.split_data(test_portion, seed)
    train_dataset = Dataset(self.x[train_idx], self.y[train_idx])
    test_dataset = Dataset(self.x[test_idx], self.y[test_idx])

    return train_dataset, test_dataset

def BaselineModel(traindata, testdata):

    Xtrain, ytrain = traindata.X, traindata.y
    Xtest, ytest = testdata.X, testdata.y
    model = LogisticRegression(penalty='l1', random_state=42, solver='saga', multi_class='multinomial')
    model.fit(Xtrain, ytrain)
    preds = model.predict(Xtest)
    acc = accuracy_score(preds, ytest)

    return acc