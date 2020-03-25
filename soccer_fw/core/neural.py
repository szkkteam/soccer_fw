# Common Python library imports
import json
from pathlib import Path

# Pip package imports
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.externals import joblib

# Internal package imports


DEFAULT_MODEL_NAME = "undefined"
DEFAULT_MODEL_VERSION = "v0_1"


class NeuralEngine:
    #-----------------------------------------------------
    # Public method declarations
    #-----------------------------------------------------
    def __init__(self, name=DEFAULT_MODEL_NAME, slug=DEFAULT_MODEL_NAME, version=DEFAULT_MODEL_VERSION, *args, **kwargs):
        self._name = name
        self._slug = slug
        self._version = version

        self._x = None
        self._y = None
        # Initialize the model
        self._model = kwargs.get('model', self.init_model(kwargs.get('params', {})))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def score(self):
        assert self._model, "NN Model is undefined!"
        assert self._x is not None, "Training dataset is not defined. First train the model"
        assert self._y is not None, "Validation dataset is not defined. First train the model"
        return self._model.score(self._x, self._y)

    @property
    def slug(self):
        return self._slug

    @property
    def name(self):
        return self._name

    def init_model(self, **model_params):
        return None

    def save(self, path, text=""):
        assert self._model, "NN Model is undefined!"
        try:
            joblib.dump(self._model, str(Path(path) / self.get_name(text) + '.sav'))
        except Exception as err:
            logger.error(err)
            raise

    def load(self, path_to_file):
        self._model = NeuralEngine.load_from(path_to_file)

    @staticmethod
    def load_from(path_to_file):
        file = Path(path_to_file)
        try:
            model = joblib.load(str(file))
        except FileNotFoundError as err:
            logger.error(err)
            raise
        return model

    def split_dataset(self, merged_dataset, percentage=0.8):
        """
        :param merged_dataset: pandas DataFrame object contains the training and validation datasets
        :param percentage: The split percentage for training and validation set. Default is 80% which means 80% training and 20% validation
        :return: (x,y) tuple, where x is the training dataset and y is the validation dataset
        """
        assert isinstance(merged_dataset, pd.DataFrame), "Input dataset must be a pandas DataFrame or numpy object"
        train_data,validation_data = np.split(merged_dataset, [int(percentage * len(merged_dataset))])
        return train_data, validation_data

    def handle_nan(self, dataset):
        return dataset.dropna(axis=0)

    def select_features(self, x_dataset):
        assert False, "Method select_features must be override by inherit class."

    def convert_to_np(self, x_dataset, y_dataset):
        x, y = None, None
        if x_dataset is not None:
            x = x_dataset.values()
        if y_dataset is not None:
            y = y_dataset.values()
        return x, y

    def preprocess(self, merged_datasets):
        return merged_datasets

    def postprocess(self, validation_data, result_array, **kwargs):
        return result_array

    def train(self, x, **kwargs):
        assert self._model, "NN Model is undefined!"

        x, y = self.select_features(x)
        x,y = self.convert_to_np(x, y)
        self._x = x
        self._y = y
        return self._model.fit(x, y, **kwargs)

    def run_train(self, datasets, **kwargs):
        #assert isinstance(dataset, pd.DataFrame), "Input dataset must be a pandas DataFrame object"
        dataset = self.handle_nan(self.preprocess(datasets))
        train_data,_ = self.split_dataset(dataset)
        return self.train(train_data, **kwargs)

    def run_predict(self, datasets, **kwargs):
        #assert isinstance(dataset, pd.DataFrame), "Input dataset must be a pandas DataFrame object"
        dataset = self.handle_nan(self.preprocess(datasets))
        _, validation_data = self.split_dataset(dataset)
        prediction =  self.predict(validation_data, **kwargs)
        return self.postprocess(validation_data, prediction, **kwargs)

    def train_step(self):
        pass

    def predict(self, x, **kwargs):
        assert self._model, "NN Model is undefined!"

        x,_ = self.select_features(x)
        x,_ = self.convert_to_np(x, None)
        prediction = self._model.predict(x, **kwargs)
        return self.postprocess(x, prediction, **kwargs)

    def statistic(self):
        pass

    def help(self):
        pass

    def get_name(self, text=""):
        text = '_' + text if len(text) > 0 else ""
        return self._slug.lower() + '_' + self._version + text

    def __repr__(self):
        return self.help()