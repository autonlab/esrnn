import os
import sys

import pandas as pd
from d3m import container, utils as d3m_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.metadata import params
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import Inputs, CallResult, Outputs, Params
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

import esrnn
from esrnn.contrib.ESRNN import ESRNN

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class ForecastingESRNNParams(params.Params):
    is_fitted: bool


class ForecastingESRNNHyperparams(hyperparams.Hyperparams):
    max_epochs = hyperparams.UniformInt(
        default=50,
        lower=0,
        upper=sys.maxsize,
        description="epochs to do on fit process",
        semantic_types=["http://schema.org/Boolean",
                        "https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
    )
    batch_size = hyperparams.UniformInt(
        default=8,
        lower=1,
        upper=10000,
        description="The batch size for RNN training",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
    )
    learning_rate = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-3,
        description='Learning rate used during training (fit).'
    )
    seasonality = hyperparams.UniformInt(
        default=30,
        lower=1,
        upper=10000,
        description="",  # TODO
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
    )
    input_size = hyperparams.UniformInt(
        default=30,
        lower=1,
        upper=10000,
        description="",  # TODO
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
    )
    output_size = hyperparams.UniformInt(
        default=60,
        lower=1,
        upper=10000,
        description="",  # TODO
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
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
            'description': "Hybrid ES-RNN models for time series forecasting",
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
        # self._esrnn = ESRNN(logger=self.logger)
        self._esrnn = ESRNN(
            max_epochs=hyperparams['max_epochs'],
            batch_size=hyperparams['batch_size'],
            learning_rate=hyperparams['learning_rate'],
            seasonality=hyperparams['seasonality'],
            input_size=hyperparams['input_size'],
            output_size=hyperparams['output_size']
        )
        self._data = None
        self._integer_time = False
        self._year_column = None

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
            # TODO
            pass
        else:
            # create a column for year
            year_column = 'year'
            count = 0
            while year_column in data.columns:
                year_column = 'year_' + str(count)
                count += count

            # create year column and add it to the grouping_keys
            data[year_column] = data[self._time_column].dt.year
            self._year_column = year_column
            self.filter_idxs.append(year_column)

            # concatenate columns in `grouping_keys` to unique_id column
            concat = data.loc[:, self.filter_idxs].apply(lambda x: '-'.join([str(v) for v in x]), axis=1)
            concat = pd.concat([concat,
                                data[year_column].astype(str),
                                data[self._time_column],
                                data[self.target_column]],
                               axis=1)
            concat.columns = ['unique_id', 'x', 'ds', 'y']

            # Series must be complete in the frequency
            concat = ForecastingESRNNPrimitive._ffill_missing_dates_per_serie(concat, 'D')

            # remove duplicates
            concat = concat[~ concat[['unique_id', 'ds']].duplicated()]

            self._data = concat

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        X_train = self._data[['unique_id', 'ds', 'x']]
        y_train = self._data[['unique_id', 'ds', 'y']]
        self._esrnn.fit(X_train, y_train, self.random_seed)
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
            # TODO
            pass
        else:
            # create year column and add it to the grouping_keys
            inputs_copy[self._year_column] = inputs_copy[self._time_column].dt.year

            # concatenate columns in `grouping_keys` to unique_id column
            concat = inputs_copy.loc[:, self.filter_idxs].apply(lambda x: '-'.join([str(v) for v in x]), axis=1)
            concat = pd.concat([concat, inputs_copy[self._time_column]], axis=1)
            concat.columns = ['unique_id', 'ds']

        X_test = concat[['unique_id', 'ds']]

        predictions = self._esrnn.predict(X_test)
        predictions = predictions['y_hat']
        output = container.DataFrame(predictions, generate_metadata=True)
        return base.CallResult(output)

    def set_params(self, *, params: Params) -> None:
        self._is_fitted = params['is_fitted']

    def get_params(self) -> Params:
        return ForecastingESRNNParams(is_fitted=self._is_fitted)

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
        df_balanced['x'] = df_balanced['x'].fillna(method='ffill')

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

        df_dates = pd.concat(df_list).reset_index(drop=True).drop('key', axis=1)[['unique_id', 'ds', 'y', 'x']]

        return df_dates
