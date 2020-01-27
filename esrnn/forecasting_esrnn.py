import os
import pandas as pd
from d3m import container, utils as d3m_utils
from d3m.metadata import params
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import Inputs, CallResult, Outputs, Params
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

import esrnn
from esrnn.contrib.esrnn import ESRNN

Input = container.DataFrame
Output = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class ForecastingESRNNParams(params.Params):
    is_fitted: bool


class ForecastingESRNNHyperparams(hyperparams.Hyperparams):
    pass


class ForecastingESRNNPrimitive(SupervisedLearnerPrimitiveBase[Input, Output, ForecastingESRNNParams,
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
                'package_uri': 'git+https://github.com/autonlab/esrnn.git@{git_commit}#egg=autonbox'.format(
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
        self._esrnn = ESRNN(logger=self.logger)  # hyperpars are specified in a .yaml file
        self._data = None

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
        self.time_column = list(data)[times[0]]

        # if datetime columns are integers, parse as # of days
        if (
                "http://schema.org/Integer"
                in inputs.metadata.query_column(times[0])["semantic_types"]
        ):
            self.integer_time = True
            data[self.time_column] = pd.to_datetime(
                data[self.time_column] - 1, unit="D"
            )
        else:
            data[self.time_column] = pd.to_datetime(
                data[self.time_column], unit="s"
            )

        # sort by time column
        data = data.sort_values(by=[self.time_column])

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

        # mark primary key
        self.key_column = 'd3mIndex'
        # self._keys = data.metadata.list_columns_with_semantic_types(
        #     (
        #         'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
        #     )
        # )
        # self._keys = [list(data)[t] for t in self._keys]
        # self.key_column = self._keys[0]

        # rename columns to the ones expected by the model
        data = data.rename(columns={
            self.key_column: 'unique_id',
            self.time_column: 'ds',
            self.target_column: 'y'
        })

        # extract required columns by the model
        self._data = data[['unique_id', 'ds', 'y']]

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        self._esrnn.fit(self._data)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        predictions = self._esrnn.predict()
        output = container.DataFrame(predictions, generate_metadata=True)
        return base.CallResult(output)

    def set_params(self, *, params: Params) -> None:
        self._is_fitted = params['is_fitted']

    def get_params(self) -> Params:
        return ForecastingESRNNParams(is_fitted=self._is_fitted)
