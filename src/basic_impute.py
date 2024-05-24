import numpy as np
import pandas as pd


def Imputation_Mean(X, seed=None):
  '''
  Implement mean imputation
  '''
  if seed != None:
    np.random.seed(seed)
  X_imputed = X.copy()
  for col in np.arange(X_imputed.shape[1]):
    col_mean = np.nanmean(X_imputed[:, col])
    X_imputed[np.isnan(X_imputed[:, col]), col] = col_mean
  return X_imputed

def Imputation_Mode(X, seed=None):
  '''
  Implement mode imputation
  '''
  if seed != None:
    np.random.seed(seed)
  X_imputed = X.copy()
  for col in np.arange(X_imputed.shape[1]):
    unique_values, counts = np.unique(X_imputed[:,col], return_counts=True)
    reverse_sorted_counts = np.argsort(counts)[::-1]
    col_mode = unique_values[0] if unique_values[0] != np.nan else unique_values[1]
    X_imputed[np.isnan(X_imputed[:, col]), col] = col_mode
  return X_imputed
