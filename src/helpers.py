import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD

def EncodeLabels(labels):
  all_labels = np.unique(labels)
  encoder = LabelEncoder()
  encoder.fit(all_labels)
  encoded = encoder.transform(labels)
  return encoded

def Implement_Random_Masking(data, proportion1, proportion2:float=None, axis:int=None, pseudo:bool=False, seed:int=174):
  '''
  This function takes X and the masking portion and return the new X with the
  same shape with missing the given portion of random data.

  Args:
  - data: full data
  - proportion1: float
  - proportion2: float 
  - axis: which axis to apply first if you want to mask out a given portion of the rows or columns.
  - pseudo: if True, the percentage of nans will always be exactly the same given the same dataset and proportions.
      if False, then the percentage of nans will be approximately proportion * dataset size
  - seed: int
  Returns:
  - masked_data: same shape as given data, missing given portion of data
  '''
  rng = np.random.default_rng(seed)
  r, c = data.shape
  mask = np.zeros((r,c))
  if axis is None:
    idxs = np.array([(i,j)for i in range(r) for j in range(c)])
    if pseudo:
      # Pseudorandom. The number of masked values is constant.
      idxs = rng.choice(idxs, size=round(proportion1*r*c), replace=False)
    else:
      # True random.
      idxs = idxs[rng.random(r*c) < proportion1]
  else:
    if proportion2 is None:
      warnings.warn('Using proportion1 as proportion2 since no proportion2 was specified.')
      proportion2=proportion1
    
    if axis == 0:
      ridxs = rng.choice(np.arange(r),size=round(proportion1*r),replace=False)
      if pseudo:
        # Pseudorandom. The number of masked values is constant.
        idxs = np.array([(i,j) for i in ridxs for j in rng.choice(np.arange(c),size=round(proportion2*c),replace=False)])
      else:
        # True random.
        idxs = np.array([(i,j) for i in ridxs for j in np.arange(c)[rng.random(c)<proportion2]])
    elif axis == 1:
      cidxs = rng.choice(np.arange(c),size=round(proportion1*c),replace=False)
      if pseudo:
        # Pseudorandom. The number of masked values is constant.
        idxs = np.array([(i,j) for j in cidxs for i in rng.choice(np.arange(r),size=round(proportion2*r),replace=False)])
      else:
        # True random.
        idxs = np.array([(i,j) for j in cidxs for i in np.arange(r)[rng.random(r)<proportion2]])
    else:
      raise Exception('axis must be 0 or 1')
    
  mask[idxs[:,0], idxs[:,1]]=1
  mask = mask.astype(bool)
  masked_data = data.copy()
  masked_data[mask] = np.nan    
  return masked_data

def HighestVarianceFeatures(data, feature_proportion):
  '''
  This function takes data and the proportion of features to keep and returns a new matrix with the proportion of
  features with highest variance.
  Input:
  data (np.ndarray)
  feature_proportion (float)

  Returns:
  the data with reduced number of features
  '''
  variances = np.array([np.var(data[:, col]) for col in range(data.shape[1])])

  n_features = round(feature_proportion*data.shape[1])

  indexes = np.argsort(variances)[-n_features:]
  explained_ratio = np.sum(variances[indexes])/np.sum(variances)
  print(f'Percentage of explained variance is: {explained_ratio}')

  return data[:, indexes]

def HighestVarianceDecomposition(data, feature_proportion, method: str='pca'):
  '''
  This function takes data and the proportion of features to keep and returns a new matrix with the proportion of
  pca features with highest variance.
  Input:
  data (np.ndarray)
  feature_proportion (float)
  method (str): "pca" or "svd"

  Returns:
  the data with reduced number of features
  '''
  n_features = round(feature_proportion*data.shape[1])

  if n_features >= min(data.shape):
    raise Exception('Number of decomposed features cannot exceed min(nrows, ncols) '+
                    'where nrows and ncols are the number rows and columns of data')
  
  if method.lower() == 'pca':
    decomposer = PCA(n_components=n_features)
  elif method.lower() == 'svd':
    decomposer = TruncatedSVD(n_components=n_features)

  decomposer.fit(data)
  explained_ratio = np.sum(decomposer.explained_variance_ratio_)
  print(f'Percentage of explained variance is: {explained_ratio}')
  return decomposer.transform(data)
