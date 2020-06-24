import os
import sys
from typing import Any

import pandas as pd
import torch
from d3m import container, utils as d3m_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.metadata import params
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import Inputs, CallResult, Outputs, Params
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

import esrnn
from esrnn.contrib.ESRNN_auto import ESRNN

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class ForecastingESRNNParams(params.Params):
    is_fitted: bool
    time_column: Any
    integer_time: Any
    filter_idxs: Any
    y_mean: Any
    esrnn: Any


class ForecastingESRNNHyperparams(hyperparams.Hyperparams):
    auto_tune = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Allow ESRNN to automatically tune the hyperparameters. You must still specify output_size.",
    )
    max_epochs = hyperparams.UniformInt(
        default=15,
        lower=0,
        upper=sys.maxsize,
        description="Maximum number of complete passes to train data during fit",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    freq_of_test = hyperparams.UniformInt(
        default=50,
        lower=0,
        upper=sys.maxsize,
        description="period for the diagnostic evaluation of the model.",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    learning_rate = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-3,
        description='Size of the stochastic gradient descent steps'
    )
    lr_decay = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=0.9,
        description='The gamma parameter of the RNN scheduler to shrink the learning rate.'
    )
    lr_scheduler_step_size = hyperparams.UniformInt(
        default=9,
        lower=1,
        upper=10000,
        description="This step_size is the period for each learning rate decay",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    per_series_lr_multip = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1.0,
        description='Multiplier for per-series parameters smoothing and initial seasonalities learning rate'
    )
    gradient_eps = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-8,
        description='term added to the Adam optimizer denominator to improve numerical stability'
    )
    gradient_clipping_threshold = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=20,
        description='max norm of gradient vector, with all parameters treated as a single vector'
    )
    rnn_weight_decay = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=0,
        description='parameter to control classic L2/Tikhonov regularization of the rnn parameters'
    )
    noise_std = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-3,
        description='standard deviation of white noise added to input during fit to avoid the model from memorizing '
                    'the train data '
    )
    level_variability_penalty = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=80,
        description='this parameter controls the strength of the penalization to the wigglines of the level vector, '
                    'induces smoothness in the output '
    )
    training_percentile = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=50,
        description='To reduce the model\'s tendency to over estimate, the training_percentile can be set to fit a smaller value through the Pinball Loss.'
                    'controls for the value predicted, when forecasting point value, the forecast is the median, '
                    'so percentile=50. '
    )
    batch_size = hyperparams.UniformInt(
        default=1,
        lower=1,
        upper=10000,
        description="The batch size for RNN training",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    batch_size_test = hyperparams.UniformInt(
        default=64,
        lower=1,
        upper=10000,
        description="The batch size for RNN test. We separated this parameter since this batch size can be "
                    "considerably larger than the train batch. It only affects the time it takes to perform "
                    "predictions",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    seasonality = hyperparams.List(
        elements=hyperparams.Hyperparameter[int](1),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=[],
        description="The main frequency of the time series. The value should between 1 and 13. Quarterly 4, Daily 7, "
                    "Monthly 12",
    )
    frequency = hyperparams.Hyperparameter[str](
        default="D",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A number of string aliases are given to useful common time series frequencies. "
                    "See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases for a list of frequency aliases",
    )
    input_size = hyperparams.UniformInt(
        default=4,
        lower=1,
        upper=10000,
        description="input size of the recursive neural network, usually a multiple of seasonality",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    output_size = hyperparams.UniformInt(
        default=60,
        lower=1,
        upper=10000,
        description="The forecast horizon of the recursive neural network, usually multiple of seasonality. The "
                    "forecast horizon is the number of periods to forecast.",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    exogenous_size = hyperparams.UniformInt(
        default=60,
        lower=1,
        upper=10000,
        description="size of one hot encoded categorical variable, invariannt per time series of the panel",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
    )
    cell_type = hyperparams.Enumeration(
        default="LSTM",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["LSTM", "GRU", "RNN", "ResLSTM", "AttentiveLSTM"],
        description="Type of RNN cell, available GRU, LSTM, RNN, ResidualLSTM",
    )
    state_hsize = hyperparams.UniformInt(
        default=40,
        lower=1,
        upper=10000,
        description="dimension of hidden state of the recursive neural network",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
    )
    dilations = hyperparams.List(
        elements=hyperparams.List(
            elements=hyperparams.Hyperparameter[int](1),
            default=[],
        ),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=[[1, 7], [28]],
        description='a list of list of ints, each list represents one chunk of Dilated LSTMS, connected in standard '
                    'ResNet fashion'
    )
    add_nl_layer = hyperparams.UniformBool(
        default=False,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="whether to insert a tanh() layer between the RNN stack and the linear adaptor (output) layers",
    )
    data_augmentation = hyperparams.UniformBool(
        default=False,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="True to turn on data augmentation support",
    )
    max_periods = hyperparams.UniformInt(
        default=20,
        lower=0,
        upper=sys.maxsize,
        description="The max number of periods (one period is one season as specified in the other hyperparameters)",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    device = hyperparams.Hyperparameter[str](
        default="cpu",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Specify the device, such as cpu, cuda, cuda:0. We recommend using CPU. It fallbacks to "
                    "CPU if GPU is not available",
    )


class ForecastingESRNNPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, ForecastingESRNNParams,
                                                               ForecastingESRNNHyperparams]):
    """
    Hybrid ES-RNN models for time series forecasting
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '098afc89-da5f-4bf4-9298-dcd39406354c',
            'version': '0.1.0',
            "name": "Hybrid ES-RNN models for time series forecasting",
            'description': "Pytorch Implementation of the M4 time series forecasting competition winner. Proposed by "
                           "Smyl. The model uses a hybrid approach of Machine Learning and statistical methods by "
                           "combining recursive neural networks to model a common trend with shared parameters across "
                           "series, and multiplicative Holt-Winter exponential smoothing.",
            'python_path': 'd3m.primitives.time_series_forecasting.esrnn.RNN',
            'source': {
                'name': esrnn.__author__,
                'uris': ['https://github.com/autonlab/esrnn'],
                'contact': 'mailto:donghanw@cs.cmu.edu'
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/autonlab/esrnn.git@{git_commit}#egg=esrnn'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.RECURRENT_NEURAL_NETWORK,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
        },
    )

    def __init__(self, *, hyperparams: ForecastingESRNNHyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._is_fitted = False

        self._device = 'cpu' if not torch.cuda.is_available() or hyperparams['device'] == 'cpu' else hyperparams['device']
        print("Use " + self._device)
        self.logger.info("Use " + self._device)

        if hyperparams['auto_tune']:
            self._esrnn = ESRNN(
                output_size=hyperparams['output_size'],
                device=self._device,
                auto_tune=hyperparams['auto_tune']
            )
        else:
            self._esrnn = ESRNN(
                max_epochs=hyperparams['max_epochs'],
                batch_size=hyperparams['batch_size'],
                batch_size_test=hyperparams['batch_size_test'],
                data_augmentation=hyperparams['data_augmentation'],
                learning_rate=hyperparams['learning_rate'],
                lr_scheduler_step_size=hyperparams['lr_scheduler_step_size'],
                lr_decay=hyperparams['lr_decay'],
                per_series_lr_multip=hyperparams['per_series_lr_multip'],
                gradient_eps=hyperparams['gradient_eps'],
                gradient_clipping_threshold=hyperparams['gradient_clipping_threshold'],
                rnn_weight_decay=hyperparams['rnn_weight_decay'],
                noise_std=hyperparams['noise_std'],
                level_variability_penalty=hyperparams['level_variability_penalty'],
                training_percentile=hyperparams['training_percentile'],
                cell_type=hyperparams['cell_type'],
                state_hsize=hyperparams['state_hsize'],
                dilations=hyperparams['dilations'],
                add_nl_layer=hyperparams['add_nl_layer'],
                seasonality=hyperparams['seasonality'],
                input_size=hyperparams['input_size'],
                output_size=hyperparams['output_size'],
                frequency=hyperparams['frequency'],
                max_periods=hyperparams['max_periods'],
                # random_seed=random_seed,  # FIXME pipelines are tuned on ESRNN's default seed
                device=self._device,
                auto_tune=hyperparams['auto_tune']
            )

        self._data = None
        self._integer_time = False
        self._year_column = None
        self._constant = 1  # the constant term to avoid nan
        self._y_mean = 0  # the mean of the target variable in the training data

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        data = inputs.horizontal_concat(outputs)
        data = data.copy()

        # mark datetime column
        times = data.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
                "http://schema.org/DateTime",
            )
        )
        if len(times) != 1:
            raise ValueError(
                f"There are {len(times)} indices marked as datetime values. Please only specify one"
            )
        self._time_column = list(data)[times[0]]

        # if datetime columns are integers, parse as # of days
        if (
                "http://schema.org/Integer"
                in inputs.metadata.query_column(times[0])["semantic_types"]
        ):
            self._integer_time = True
            data[self._time_column] = pd.to_datetime(
                data[self._time_column] - 1, unit="D"
            )
        else:
            data[self._time_column] = pd.to_datetime(
                data[self._time_column], unit="s"
            )

        # sort by time column
        data = data.sort_values(by=[self._time_column])

        # mark key and grp variables
        self.key = data.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
        )

        # mark target variables
        self._targets = data.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                "https://metadata.datadrivendiscovery.org/types/Target",
            )
        )
        self._target_types = [
            "i"
            if "http://schema.org/Integer"
               in data.metadata.query_column(t)["semantic_types"]
            else "c"
            if "https://metadata.datadrivendiscovery.org/types/CategoricalData"
               in data.metadata.query_column(t)["semantic_types"]
            else "f"
            for t in self._targets
        ]
        self._targets = [list(data)[t] for t in self._targets]

        self.target_column = self._targets[0]

        # see if 'GroupingKey' has been marked
        # otherwise fall through to use 'SuggestedGroupingKey'
        grouping_keys = data.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey"
        )
        suggested_grouping_keys = data.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
        )
        if len(grouping_keys) == 0:
            grouping_keys = suggested_grouping_keys
            drop_list = []
        else:
            drop_list = suggested_grouping_keys

        grouping_keys_counts = [
            data.iloc[:, key_idx].nunique() for key_idx in grouping_keys
        ]
        grouping_keys = [
            group_key
            for count, group_key in sorted(zip(grouping_keys_counts, grouping_keys))
        ]
        self.filter_idxs = [list(data)[key] for key in grouping_keys]

        # drop index
        data.drop(
            columns=[list(data)[i] for i in drop_list + self.key], inplace=True
        )

        # check whether no grouping keys are labeled
        if len(grouping_keys) == 0:
            concat = pd.concat([data[self._time_column], data[self.target_column]], axis=1)
            concat.columns = ['ds', 'y']
            concat['unique_id'] = 'series1'  # We have only one series
        else:
            # concatenate columns in `grouping_keys` to unique_id column
            concat = data.loc[:, self.filter_idxs].apply(lambda x: ' '.join([str(v) for v in x]), axis=1)
            concat = pd.concat([concat,
                                data[self._time_column],
                                data[self.target_column]],
                               axis=1)
            concat.columns = ['unique_id', 'ds', 'y']

        if len(grouping_keys):
            # Series must be complete in the frequency
            concat = ForecastingESRNNPrimitive._ffill_missing_dates_per_serie(concat, self.hyperparams['frequency'])

        # remove duplicates
        concat = concat.drop_duplicates(['unique_id', 'ds'])

        self._data = concat

        self._y_mean = self._data['y'].mean()

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        X_train = self._data[['unique_id', 'ds']]
        X_train['x'] = '1'
        y_train = self._data[['unique_id', 'ds', 'y']]
        y_train['y'] += self._constant
        self._esrnn.fit(X_train, y_train)
        self._is_fitted = True

        return base.CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._is_fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        inputs_copy = inputs.copy()

        # if datetime columns are integers, parse as # of days
        if self._integer_time:
            inputs_copy[self._time_column] = pd.to_datetime(
                inputs_copy[self._time_column] - 1, unit="D"
            )
        else:
            inputs_copy[self._time_column] = pd.to_datetime(
                inputs_copy[self._time_column], unit="s"
            )

        # find marked 'GroupingKey' or 'SuggestedGroupingKey'
        grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey"
        )
        suggested_grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
        )
        if len(grouping_keys) == 0:
            grouping_keys = suggested_grouping_keys
        else:
            inputs_copy = inputs_copy.drop(columns=[list(inputs_copy)[i] for i in suggested_grouping_keys])

        # check whether no grouping keys are labeled
        if len(grouping_keys) == 0:
            concat = pd.concat([inputs_copy[self._time_column]], axis=1)
            concat.columns = ['ds']
            concat['unique_id'] = 'series1'  # We have only one series
        else:
            # concatenate columns in `grouping_keys` to unique_id column
            concat = inputs_copy.loc[:, self.filter_idxs].apply(lambda x: ' '.join([str(v) for v in x]), axis=1)
            concat = pd.concat([concat, inputs_copy[self._time_column]], axis=1)
            concat.columns = ['unique_id', 'ds']

        X_test = concat[['unique_id', 'ds']]

        predictions = self._esrnn.predict(X_test)
        predictions['y_hat'] -= self._constant
        predictions['y_hat'] = self._fillna(predictions['y_hat'])
        output = container.DataFrame(predictions['y_hat'], generate_metadata=True)
        return base.CallResult(output)

    def set_params(self, *, params: Params) -> None:
        self._is_fitted = params['is_fitted']
        self._time_column = params['time_column']
        self._integer_time = params['integer_time']
        self.filter_idxs = params['filter_idxs']
        self._y_mean = params['y_mean']
        self._esrnn = params['esrnn']

    def get_params(self) -> Params:
        return ForecastingESRNNParams(is_fitted=self._is_fitted,
                                      time_column=self._time_column,
                                      integer_time=self._integer_time,
                                      filter_idxs=self.filter_idxs,
                                      y_mean=self._y_mean,
                                      esrnn=self._esrnn)

    @staticmethod
    def _ffill_missing_dates_particular_serie(serie, min_date, max_date, freq):
        date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
        unique_id = serie['unique_id'].unique()
        df_balanced = pd.DataFrame({'ds': date_range, 'key': [1] * len(date_range), 'unique_id': unique_id[0]})

        # Check balance
        check_balance = df_balanced.groupby(['unique_id']).size().reset_index(name='count')
        assert len(set(check_balance['count'].values)) <= 1
        df_balanced = df_balanced.merge(serie, how="left", on=['unique_id', 'ds'])

        df_balanced['y'] = df_balanced['y'].fillna(method='ffill')

        return df_balanced

    @staticmethod
    def _ffill_missing_dates_per_serie(df, freq="D", fixed_max_date=None):
        """Receives a DataFrame with a date column and forward fills the missing gaps in dates, not filling dates before
        the first appearance of a unique key

        Parameters
        ----------
        df: DataFrame
            Input DataFrame
        key: str or list
            Name(s) of the column(s) which make a unique time series
        date_col: str
            Name of the column that contains the time column
        freq: str
            Pandas time frequency standard strings, like "W-THU" or "D" or "M"
        numeric_to_fill: str or list
            Name(s) of the columns with numeric values to fill "fill_value" with
        """
        if fixed_max_date is None:
            df_max_min_dates = df[['unique_id', 'ds']].groupby('unique_id').agg(['min', 'max']).reset_index()
        else:
            df_max_min_dates = df[['unique_id', 'ds']].groupby('unique_id').agg(['min']).reset_index()
            df_max_min_dates['max'] = fixed_max_date

        df_max_min_dates.columns = df_max_min_dates.columns.droplevel()
        df_max_min_dates.columns = ['unique_id', 'min_date', 'max_date']

        df_list = []
        for index, row in df_max_min_dates.iterrows():
            df_id = df[df['unique_id'] == row['unique_id']]
            df_id = ForecastingESRNNPrimitive._ffill_missing_dates_particular_serie(df_id, row['min_date'],
                                                                                    row['max_date'], freq)
            df_list.append(df_id)

        df_dates = pd.concat(df_list).reset_index(drop=True).drop('key', axis=1)[['unique_id', 'ds', 'y']]

        return df_dates

    def _fillna(self, series):
        if series.isnull().any():
            # self.logger.warning("The prediction contains NAN. Fill with mean of training data. You may want to "
            #                     "increase output_size.")
            tofill = series.mean()  # use the prediction mean if possible. Otherwise use the mean of the training data.
            if pd.isna(tofill):
                # self.logger.warn('The predictions are all NAN')
                tofill = self._y_mean
            return series.fillna(tofill)
        return series
