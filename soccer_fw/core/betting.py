# Common Python library imports
import json
from pathlib import Path

# Pip package imports
from loguru import logger
import numpy as np
import pandas as pd

# Internal package imports
from soccer_fw.utils import fulltime_result_tags, common_column_names

DEFAULT_BETTING_MODEL = "undefined"
DEFAULT_BETTING_MODEL_VERSION = "v0_1"



class BettingEngine():

    config = {
        'stake_limits': np.array([0, np.inf]),
        'odd_limits' : np.array([0, np.inf]),
        'stake': 5,
        'post_modifier': lambda stake, **kwargs: stake
    }

    def __init__(self, name=DEFAULT_BETTING_MODEL, slug=DEFAULT_BETTING_MODEL, version=DEFAULT_BETTING_MODEL_VERSION, *args, **kwargs):
        self._name = name
        self._slug = slug
        self._ver = version

        self._stakes = []
        self._bet_on = 0
        self._skipped = 0

        self._config = { **BettingEngine.config , **kwargs.get('params', {})}

    @property
    def statistic(self):
        return pd.DataFrame({'Stakes': self._stakes})


    def calculate(self, *args, **kwargs):
        stake = 0
        row = kwargs.get('row', None)
        history = kwargs.get('history', None)
        statistic = kwargs.get('statistic', None)

        bankroll = kwargs.get('bankroll', None)
        prediction = kwargs.get('prediction', None)
        probability = kwargs.get('probability', None)
        odd = kwargs.get('odd', None)
        if row is not None:
            composed_return = BettingEngine._get_row_elements(row)
            bankroll = composed_return['bankroll'] if bankroll is None else bankroll
            prediction = composed_return['prediction'] if prediction is None else prediction
            probability = composed_return['probability'] if probability is None else probability
            odd = composed_return['odd'] if odd is None else odd

        assert bankroll is not None, "bankroll input parameter is mandatory but missing."
        if history is not None:
            if not self.history_check(history):
                self._skipped += 1
                return stake
        if prediction is not None:
            if not self.prediction_check( (prediction, probability) ):
                self._skipped += 1
                return stake
        if odd is not None:
            if not self.odd_check(odd):
                self._skipped += 1
                return stake

        # Calculate the stake
        stake = self.calculate_stake(bankroll=bankroll, prediction=prediction, probability=probability, odd=odd, statistic=statistic)

        # If the post modifier is callable, call it. Otherwise assign it to the stake and override the original stake
        if hasattr(self._config['post_modifier'], '__call__'):
            stake = self._config['post_modifier'](stake, bankroll=bankroll, prediction=prediction, probability=probability, odd=odd, statistic=statistic)
        else:
            stake = self._config['post_modifier']

        if stake > 0:
            self._bet_on += 1
            self._stakes.append(stake)
        else:
            self._skipped += 1
        return stake

    def prediction_check(self, pred_prob):
        return True

    def odd_check(self, odd):
        """
        Return with True or False if the odds for the prediction result is within the range or not.
        :param odd: Odd of the predicted result. (Assumption: We are betting on the prediction and not against it)
        :return: True or False
        """
        return (odd < self._config['odd_limits'][1] and odd > self._config['odd_limits'][0])

    def history_check(self, history_df):
        return True

    def calculate_stake(self, bankroll, **kwargs):
        return self._config['stake']

    def save(self, path, text=""):
        try:
            json.dump(self._config, str(Path(path) / self.get_name(text) + '.sav'))
        except Exception as err:
            logger.error(err)
            raise

    def load(self, path_to_file):
        pass

    def get_name(self, text=""):
        text = '_' + text if len(text) > 0 else ""
        return self._slug.lower() + '_' + self._ver + text

    @staticmethod
    def is_draw(prediction):
        if prediction in fulltime_result_tags['Draw']:
            return True
        return False

    @staticmethod
    def is_home(prediction):
        if prediction in fulltime_result_tags['Home']:
            return True
        return False

    @staticmethod
    def is_away(prediction):
        if prediction in fulltime_result_tags['Away']:
            return True
        return False

    @staticmethod
    def _get_row_elements(row):
        composed_values= {}
        for name, column in common_column_names:
            try:
                composed_values[name] = row[column]
            except Exception:
                pass
        return composed_values