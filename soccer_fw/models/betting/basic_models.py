# Common Python library imports

# Pip package imports
from loguru import logger
import numpy as np

# Internal package imports
from soccer_fw.core.betting import BettingEngine


class FixStakeOnlyDraw(BettingEngine):

    name = "Fix Stake Model Only Draw"
    slug = "fix_stake_model_only_draw"
    version = "v0_1"

    default_config = {
        'stake': 5,
        'percentage': 0.05
    }

    def __init__(self, *args, **kwargs):
        kwargs['params'] = { **FixStakeOnlyDraw.default_config, **kwargs.get('params',{}) }

        m_kwargs = { **{
            'name': FixStakeOnlyDraw.name,
            'slug': FixStakeOnlyDraw.slug,
            'version': FixStakeOnlyDraw.version
        }, **kwargs}

        super(FixStakeOnlyDraw, self).__init__(*args, **m_kwargs)

    def prediction_check(self, pred_prob):
        return BettingEngine.is_draw(pred_prob[0])

    def calculate_stake(self, bankroll, **kwargs):
        return bankroll * self._config['percentage']

class KellyStakeOnlyDraw(BettingEngine):

    name = "Kelly Stake Model Only Draw"
    slug = "kelly_stake_model_only_draw"
    version = "v0_1"

    default_config = {
        'percent_limit': np.array([0.00, 1.0]),
        'daily_fix_bankroll': True,
    }

    def __init__(self, *args, **kwargs):
        kwargs['params'] = { **KellyStakeOnlyDraw.default_config, **kwargs.get('params',{}) }

        m_kwargs = { **{
            'name': KellyStakeOnlyDraw.name,
            'slug': KellyStakeOnlyDraw.slug,
            'version': KellyStakeOnlyDraw.version
        }, **kwargs}

        super(KellyStakeOnlyDraw, self).__init__(*args, **m_kwargs)

    def prediction_check(self, pred_prob):
        return BettingEngine.is_draw(pred_prob[0])

    def calculate_stake(self, bankroll, **kwargs):
        probability = kwargs.get('probability', None)
        odd = kwargs.get('odd', None)
        stats = kwargs.get('statistic')

        try:
            if self._config['daily_fix_bankroll']:
                pending_stakes = stats._pending_stakes['Stake'].sum()
        except Exception:
            pending_stakes = 0

        assert odd is not None and probability is not None, "odd and probability are mandatory input parameters."

        # Correction if already a percentage
        probability = float(probability / 100.0) if float(probability) > 1.0 else float(probability)
        # Stake percentage
        stake_percent = (((float(odd) - 1.0) * probability) - (1.0 - probability)) / (float(odd) - 1.0)
        # Adjust the stake percent according to the configured min/max
        stake_percent = max(self._config['percent_limit'][0], min(stake_percent, self._config['percent_limit'][1]))

        return (pending_stakes + bankroll) * stake_percent

class BankrollPercentOnWeek(BettingEngine):

    name = "Bankroll Percentage Over Week Only Draw"
    slug = "bankroll_percentage_over_week_only_draw"
    version = "v0_1"

    default_config = {
        'stake': 5,
        'percentage': 0.5,
        'matches_per_week' : 10,
    }

    def __init__(self, *args, **kwargs):
        kwargs['params'] = { **BankrollPercentOnWeek.default_config, **kwargs.get('params',{}) }

        self._steps = 0
        self._first_week_bankroll = 0

        m_kwargs = { **{
            'name': BankrollPercentOnWeek.name,
            'slug': BankrollPercentOnWeek.slug,
            'version': BankrollPercentOnWeek.version
        }, **kwargs}

        super(BankrollPercentOnWeek, self).__init__(*args, **m_kwargs)

    def prediction_check(self, pred_prob):
        return BettingEngine.is_draw(pred_prob[0])

    def calculate_stake(self, bankroll, **kwargs):
        if self._steps == 0:
            self._first_week_bankroll = bankroll
            self._steps += 1
        if self._steps < self._config['matches_per_week'] :
            self._steps += 1
        else:
            self._steps = 0

        return (self._first_week_bankroll * self._config['percentage']) / self._config['matches_per_week']

