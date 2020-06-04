import os
import time

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from pathlib import Path
from esrnn.contrib.utils.config import ModelConfig
from esrnn.contrib.utils.ESRNN import _ESRNN
from esrnn.contrib.utils.losses import SmylLoss, PinballLoss
from esrnn.contrib.utils.data import Iterator


class ESRNN(object):
  """ Exponential Smoothing Recursive Neural Network.

  Pytorch Implementation of the M4 time series forecasting competition winner.
  Proposed by Smyl. The model uses a hybrid approach of Machine Learning and
  statistical methods by combining recursive neural networks to model a common
  trend with shared parameters across series, and multiplicative Holt-Winter
  exponential smoothing.

  Parameters
  ----------
  max_epochs: int
    maximum number of complete passes to train data during fit
  learning_rate: float
    size of the stochastic gradient descent steps
  lr_scheduler_step_size: int
    this step_size is the period for each learning rate decay
  data_augmentation: bool
    performs data augmentation, in which unique_id is concatenated
    with the year in ds column to increase number of series
  per_series_lr_multip: float
    multiplier for per-series parameters smoothing and initial
    seasonalities learning rate (default 1.0)
  gradient_eps: float
    term added to the Adam optimizer denominator to improve
    numerical stability (default: 1e-8)
  gradient_clipping_threshold: float
    max norm of gradient vector, with all parameters treated
    as a single vector
  rnn_weight_decay: float
    parameter to control classic L2/Tikhonov regularization
    of the rnn parameters
  noise_std: float
    standard deviation of white noise added to input during
    fit to avoid the model from memorizing the train data
  level_variability_penalty: float
    this parameter controls the strength of the penalization
    to the wigglines of the level vector, induces smoothness
    in the output
  training_percentile: float
    To reduce the model's tendency to over estimate, the
    training_percentile can be set to fit a smaller value
    through the Pinball Loss.
  batch_size: int
    number of training examples for the stochastic gradient steps
  seasonality: int list
    list of seasonalities of the time series
    Hourly [24, 168], Daily [7], Weekly [52], Monthly [12],
    Quarterly [4], Yearly [].
  input_size: int
    input size of the recursive neural network, usually a
    multiple of seasonality
  output_size: int
    output_size or forecast horizon of the recursive neural
    network, usually multiple of seasonality
  random_seed: int
    random_seed for pseudo random pytorch initializer and
    numpy random generator
  exogenous_size: int
    size of one hot encoded categorical variable, invariannt
    per time series of the panel
  min_inp_seq_length: int
    description
  max_periods: int
    Parameter to chop longer series, to last max_periods,
    max e.g. 40 years
  cell_type: str
    Type of RNN cell, available GRU, LSTM, RNN, ResidualLSTM.
  state_hsize: int
    dimension of hidden state of the recursive neural network
  dilations: int list
    each list represents one chunk of Dilated LSTMS, connected in
    standard ResNet fashion
  add_nl_layer: bool
    whether to insert a tanh() layer between the RNN stack and the
    linear adaptor (output) layers
  device: str
    pytorch device either 'cpu' or 'cuda'
  Notes
  -----
  **References:**
  `M4 Competition Conclusions
  <https://rpubs.com/fotpetr/m4competition>`__
  `Original Dynet Implementation of ESRNN
  <https://github.com/M4Competition/M4-methods/tree/master/118%20-%20slaweks17>`__
  """
  def __init__(self, max_epochs=15, batch_size=1, batch_size_test=64,
               learning_rate=1e-3, lr_scheduler_step_size=9, lr_decay=0.9, data_augmentation=False,
               per_series_lr_multip=1.0, gradient_eps=1e-8, gradient_clipping_threshold=20,
               rnn_weight_decay=0, noise_std=0.001, level_variability_penalty=80,
               training_percentile=50, cell_type='LSTM',
               state_hsize=40, dilations=[[1, 2], [4, 8]],
               add_nl_layer=False, seasonality=[4], input_size=4, output_size=8,
               frequency='D', max_periods=20, random_seed=1,
               device='cpu', root_dir='./'):
    super(ESRNN, self).__init__()
    self.mc = ModelConfig(max_epochs=max_epochs, batch_size=batch_size, batch_size_test=batch_size_test,
                          learning_rate=learning_rate, lr_scheduler_step_size=lr_scheduler_step_size, lr_decay=lr_decay,
                          data_augmentation=data_augmentation, per_series_lr_multip=per_series_lr_multip,
                          gradient_eps=gradient_eps, gradient_clipping_threshold=gradient_clipping_threshold,
                          rnn_weight_decay=rnn_weight_decay, noise_std=noise_std,
                          level_variability_penalty=level_variability_penalty, training_percentile=training_percentile,
                          cell_type=cell_type,
                          state_hsize=state_hsize, dilations=dilations, add_nl_layer=add_nl_layer,
                          seasonality=seasonality, input_size=input_size, output_size=output_size,
                          frequency=frequency, max_periods=max_periods, random_seed=random_seed,
                          device=device, root_dir=root_dir)
    self._fitted = False

  def train(self, dataloader, max_epochs, shuffle=True, verbose=True):

    if verbose:
        print(15*'='+' Training ESRNN  ' + 15*'=' + '\n')

    # Optimizers
    self.es_optimizer = optim.Adam(params=self.esrnn.es.parameters(),
                              lr=self.mc.learning_rate*self.mc.per_series_lr_multip,
                              betas=(0.9, 0.999), eps=self.mc.gradient_eps)

    self.es_scheduler = StepLR(optimizer=self.es_optimizer,
                          step_size=self.mc.lr_scheduler_step_size,
                          gamma=0.9)

    self.rnn_optimizer = optim.Adam(params=self.esrnn.rnn.parameters(),
                              lr=self.mc.learning_rate,
                              betas=(0.9, 0.999), eps=self.mc.gradient_eps,
                              weight_decay=self.mc.rnn_weight_decay)

    self.rnn_scheduler = StepLR(optimizer=self.rnn_optimizer,
                          step_size=self.mc.lr_scheduler_step_size,
                          gamma=self.mc.lr_decay)

    # Loss Functions
    train_tau = self.mc.training_percentile / 100
    train_loss = SmylLoss(tau=train_tau, level_variability_penalty=self.mc.level_variability_penalty)

    for epoch in range(max_epochs):
      self.esrnn.train()
      start = time.time()
      if shuffle:
        dataloader.shuffle_dataset(random_seed=epoch)
      losses = []
      for j in range(dataloader.n_batches):
        self.es_optimizer.zero_grad()
        self.rnn_optimizer.zero_grad()

        batch = dataloader.get_batch()
        windows_y, windows_y_hat, levels = self.esrnn(batch)

        # Pinball loss on normalized values
        loss = train_loss(windows_y, windows_y_hat, levels)
        losses.append(loss.data.cpu().numpy())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.esrnn.rnn.parameters(), self.mc.gradient_clipping_threshold)
        torch.nn.utils.clip_grad_norm_(self.esrnn.es.parameters(), self.mc.gradient_clipping_threshold)
        self.rnn_optimizer.step()
        self.es_optimizer.step()

      # Decay learning rate
      self.es_scheduler.step()
      self.rnn_scheduler.step()

      # Evaluation
      self.train_loss = np.mean(losses)
      if verbose:
        print("========= Epoch {} finished =========".format(epoch))
        print("Training time: {}".format(round(time.time()-start, 5)))
        print("Training loss ({} prc): {:.5f}".format(self.mc.training_percentile,
                                                      self.train_loss))

    if verbose: print('Train finished! \n')

  def get_trainable_df(self, X_df, y_df):
    unique_counts = X_df.groupby('unique_id').count().reset_index()[['unique_id','ds']]
    ids = unique_counts[unique_counts['ds'] >= self.mc.min_series_length]['unique_id'].unique()
    X_df = X_df[X_df['unique_id'].isin(ids)].reset_index(drop=True)
    y_df = y_df[y_df['unique_id'].isin(ids)].reset_index(drop=True)
    return X_df, y_df

  def fit(self, X_df, y_df, shuffle=True, verbose=False):
    # Transform long dfs to wide numpy
    assert type(X_df) == pd.core.frame.DataFrame
    assert type(y_df) == pd.core.frame.DataFrame
    assert all([(col in X_df) for col in ['unique_id', 'ds', 'x']])
    assert all([(col in y_df) for col in ['unique_id', 'ds', 'y']])

    if self.mc.data_augmentation:
      X_df['unique_id'] = X_df['unique_id'] + "_" + X_df['ds'].dt.year.astype(str)
      y_df['unique_id'] = y_df['unique_id'] + "_" + y_df['ds'].dt.year.astype(str)

    X_df, y_df = self.get_trainable_df(X_df, y_df)

    X, y = self.long_to_wide(X_df, y_df)
    assert len(X)==len(y)
    assert X.shape[1]>=3

    # Exogenous variables
    unique_categories = np.unique(X[:, 1])
    self.mc.category_to_idx = dict((word, index) for index, word in enumerate(unique_categories))
    exogenous_size = len(unique_categories)

    # Create batches (device in mc)
    self.train_dataloader = Iterator(mc=self.mc, X=X, y=y)

    # Random Seeds (model initialization)
    torch.manual_seed(self.mc.random_seed)

    # Initialize model
    n_series = self.train_dataloader.n_series
    self.instantiate_esrnn(exogenous_size, n_series)

    # Train model
    self._fitted = True
    self.train(dataloader=self.train_dataloader, max_epochs=self.mc.max_epochs,
               shuffle=shuffle, verbose=verbose)

  def instantiate_esrnn(self, exogenous_size, n_series):
    self.mc.exogenous_size = exogenous_size
    self.mc.n_series = n_series
    self.esrnn = _ESRNN(self.mc).to(self.mc.device)

  def predict(self, X_df):
    X_df = X_df.copy()
    """
        Predictions for all stored time series
    Returns:
        Y_hat_panel : array-like (n_samples, 1).
          Predicted values for models in Family for ids in Panel.
        ds: Corresponding list of date stamps
        unique_id: Corresponding list of unique_id
    """

    if self.mc.data_augmentation:
      X_df['unique_id_real'] = X_df['unique_id']
      X_df['unique_id'] = X_df['unique_id'] + "_" + X_df['ds'].dt.year.astype(str)
      unique_id_mapping = dict(zip(X_df.unique_id, X_df.unique_id_real))
      X_df = X_df.drop(columns=['unique_id_real'])

    # print(9*'='+' Predicting ESRNN ' + 9*'=' + '\n')
    assert type(X_df) == pd.core.frame.DataFrame
    assert 'unique_id' in X_df

    # Obtain unique_ids to predict
    predict_unique_idxs = X_df['unique_id'].unique()

    # Predictions for panel
    Y_hat_panel = pd.DataFrame(columns=['unique_id', 'y_hat'])

    # Imputate flag
    impute_flag = False

    for unique_id in predict_unique_idxs:
      # Corresponding train batch
      if unique_id in self.train_dataloader.sort_key['unique_id']:
        batch = self.train_dataloader.get_batch(unique_id=unique_id)

        # Prediction
        Y_hat_id = pd.DataFrame(np.zeros(shape=(self.mc.output_size, 1)), columns=["y_hat"])
        y_hat = self.esrnn.predict(batch)
        y_hat = y_hat.data.cpu().numpy()

        y_hat = y_hat.squeeze()
        Y_hat_id.iloc[:, 0] = y_hat

        # Serie prediction
        Y_hat_id["unique_id"] = unique_id
        ts = pd.date_range(start=batch.last_ds[0],
                                        periods=self.mc.output_size+1, freq=self.mc.frequency)
        Y_hat_id["ds"] = ts[1:]

        Y_hat_panel = Y_hat_panel.append(Y_hat_id, sort=False).reset_index(drop=True)
      else:
        impute_flag = True

    if 'ds' in X_df:
      Y_hat_panel = X_df.merge(Y_hat_panel, on=['unique_id', 'ds'], how='left')

    if impute_flag:
      mean_imputator = Y_hat_panel[['ds','y_hat']].groupby('ds').mean().reset_index()
      mean_imputator = mean_imputator.rename(columns={'y_hat':'y_hat_mean'})
      Y_hat_panel = Y_hat_panel.merge(mean_imputator, on=['ds'], how='left')
      Y_hat_panel['y_hat'] = Y_hat_panel['y_hat'].fillna(Y_hat_panel['y_hat_mean'])
      Y_hat_panel = Y_hat_panel.drop(columns=['y_hat_mean'])

    if self.mc.data_augmentation:
      Y_hat_panel = Y_hat_panel.replace({'unique_id':unique_id_mapping})

    return Y_hat_panel

  def long_to_wide(self, X_df, y_df):
    data = X_df.copy()
    data['y'] = y_df['y'].copy()
    sorted_ds = np.sort(data['ds'].unique())
    ds_map = {}
    for dmap, t in enumerate(sorted_ds):
        ds_map[t] = dmap
    data['ds_map'] = data['ds'].map(ds_map)
    data = data.sort_values(by=['ds_map','unique_id'])
    df_wide = data.pivot(index='unique_id', columns='ds_map')['y']

    x_unique = data[['unique_id', 'x']].groupby('unique_id').first()
    last_ds =  data[['unique_id', 'ds']].groupby('unique_id').last()
    assert len(x_unique)==len(data.unique_id.unique())
    df_wide['x'] = x_unique
    df_wide['last_ds'] = last_ds
    df_wide = df_wide.reset_index().rename_axis(None, axis=1)

    ds_cols = data.ds_map.unique().tolist()
    X = df_wide.filter(items=['unique_id', 'x', 'last_ds']).values
    y = df_wide.filter(items=ds_cols).values

    return X, y

  def get_dir_name(self, root_dir=None):
    if not root_dir:
      assert self.mc.root_dir
      root_dir = self.mc.root_dir

    data_dir = self.mc.dataset_name
    model_parent_dir = os.path.join(root_dir, data_dir)
    model_path = ['num_series_{}'.format(self.mc.num_series),
                    'lr_{}'.format(self.mc.learning_rate),
                    str(self.mc.copy)]
    model_dir = os.path.join(model_parent_dir, '_'.join(model_path))
    return model_dir