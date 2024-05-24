import numpy as np
import multiprocessing as mp
import helpers

def MSE(y, pred):
  return np.sum((y - pred)**2)

class FeatureImputationPipeline():
  def get_params(self, parameters_config):
    """
    Require:
    p_rows : the proportion of rows to mask
    p_cols : the proportion of rows to mask

    can have different imputation parameters depending on the method.

    Input : 
    paramters config : A dictionary mapping each parameter to a value.

    Output : dump the values in parameters config into self.params
    """

    self.params = {}
    for k in parameters_config:
        self.params[k] = parameters_config[k]
    return

  def __init__(self, method, parameters_config, data, loss_func = MSE):
    self.method = method
    self.get_params(parameters_config)
    self.features = data["features"]
    self.labels = data["labels"]
    self.loss = loss_func
    return

  def mask_data(self, seed):
    """
    Masks data randomly given proportion of rows and columns.
    """
    self.masked_data = helpers.Implement_Random_Masking(self.features,
                                                   self.params["p_rows"],
                                                   self.params["p_cols"],
                                                   seed)
    return

  def run_simulation(self, seed):
    """
    Given a seed, run a masking and imputing loop.
    Compute the loss given a specific loss function.

    Returns the imputed data matrix and the loss.
    """
    if np.sum(np.isnan(self.features)) == 0:
      self.mask_data(seed)
    else:
      self.masked_data = self.features
      
    res = self.method(self.masked_data, seed = seed)
    loss = self.loss(res, self.features)

    return res, loss

  def run_multiple_seeds(self, num_simuls, save_path, method_name):
    """
    Averages results of simulation across multiple
    seeds. Runs num_simuls times and returns mean loss.
    """
    prow, pcol = self.params['p_rows'], self.params['p_cols']
    if save_path[-1] != '/': save_path = save_path + '/'
    save_path = save_path + f'{method_name}_prow{100*prow:02.0f}pcol{100*pcol:02.0f}.npy'

    losses = []
    imputedData = []
    for seed in range(num_simuls):
      data, loss = self.run_simulation(seed)
      imputedData.append(data)
      losses.append(loss)
      np.save(save_path, np.array(imputedData))
      
    return np.mean(losses)

  def run_multiple_params(self, num_simuls, variables, save_dir: str=None, method_name: str=None):
    """
    Run multiple seeds for different parameter values.
    For example, can test the performance of imputation
    method as the proportion of rows increase.
    """
    mean_loss = []

    num_conditions = None
    for values in variables.values():
      if num_conditions is None:
        num_conditions = len(values)
      assert len(values) == num_conditions, "The length of the iterable for each key of variables should be identical."

    for i in range(num_conditions):
      for name, values in variables.items():
        self.params[name] = values[i]
      mean_loss.append(self.run_multiple_seeds(num_simuls, save_dir, method_name))

    return mean_loss