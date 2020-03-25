# Common Python library imports

# Pip package imports
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Internal package imports
from soccer_fw.utils import listify, fulltime_result_tags

DEFAULT_ENGINE_NAME = "default"
DEFAULT_ENGINE_VERSION = "v0_1"


class StatisticModel():

    def __init__(self, bankroll, *args, **kwargs):
        self._initial_bankroll = bankroll
        self._bankroll = bankroll
        self._v_bankroll = bankroll
        self._pending_stakes = pd.DataFrame()
        self._win_streaks = np.array([0])
        self._loose_streaks = np.array([0])
        self._profit = 0

        self._matches_won = 0
        self._total_matches = 0
        self._matches_played = 0

        self._idx = None

        self._stat_df = pd.DataFrame()

    @property
    def hitrate(self):
        return self._matches_won / self._matches_played

    @property
    def roi(self):
        return self._profit / self._stat_df['Stake'].sum()

    @property
    def dataframe(self):
        if self._idx is not None:
            return self._stat_df.set_index(self._idx)
        return self._stat_df

    @property
    def bankroll(self):
        return self._bankroll

    @property
    def profit(self):
        return self._profit


    @property
    def loose_streak(self):
        return self._loose_streaks

    @property
    def win_streak(self):
        return self._win_streaks


    def plot(self):
        # gca stands for 'get current axis'
        ax = plt.gca()
        df = pd.concat([self.dataframe, pd.DataFrame({ 'Win Streak': self.win_streak, 'Lose Streak': self.loose_streak})], axis=1)

        ls_max = df['Lose Streak'].max()
        b_maximum = df['Bankroll'].max() * 0.7 / ls_max
        df['Win Streak'] = df['Win Streak'] * b_maximum
        df['Lose Streak'] = df['Lose Streak'] * b_maximum

        df.plot(kind='line', y='Bankroll', color='blue', ax=ax)
        df.plot(kind='bar', y='Stake', color='green', ax=ax)
        df.plot(kind='bar', y='Profit', color='yellow', ax=ax)
        df.plot(kind='bar', y='Lose Streak', color='red', ax=ax)
        df.plot(kind='bar', y='Win Streak', color='grey', ax=ax)

        plt.show()

    def place_bet(self, stake, odd, win):
        """
        Place a bet. The stake will be substracted from the bankroll and added to the pending bets.
        Pending bet's has to be evaluated by the eval_bet call. The function return with the bet index
        :param stake: Amount money to bet
        :return: Index of the bet
        """
        assert (self._bankroll - stake > 0), "The bankroll cannot go to negative"

        self._pending_stakes = self._pending_stakes.append({ 'Stake': stake, 'Odd': odd, 'Win': win}, ignore_index=True)
        self._bankroll -= stake

        return len(self._pending_stakes.index) - 1

    def eval_bet(self, index, key=None):
        assert key is None or isinstance(key, tuple), "Key has to a tuple or None"
        index_row = self._pending_stakes[self._pending_stakes.index == index]
        pending_stake = self._pending_stakes[self._pending_stakes.index > index]['Stake'].sum()
        stake = float(index_row['Stake'])
        odd = float(index_row['Odd'])
        win = float(index_row['Win'])
        won_amount = 0

        # Modify the statistic only when the match was played
        if stake > 0:
            self._matches_played += 1

            if win > 0.0:
                won_amount = float(stake) * float(odd)
                self._bankroll += won_amount
                # Store statistics
                self._profit += (won_amount - stake)
                self._win_streaks = np.append(self._win_streaks, self._win_streaks[-1] + 1)
                self._loose_streaks = np.append(self._loose_streaks, 0)
                self._matches_won += 1
            else:
                self._profit -= stake
                # Store statistics
                self._loose_streaks = np.append(self._loose_streaks, self._loose_streaks[-1] + 1)
                self._win_streaks = np.append(self._win_streaks, 0)

        stat = {
            'Bankroll': self.bankroll,
            'Stake': stake,
            'Odd': odd,
            'Virtual Bankroll': self.bankroll + pending_stake,
            'Pending Stake': pending_stake,
            'Profit': won_amount,
            'Win': 'Won' if win > 0.0 else 'Lost'
        }
        if key is not None:
            self._idx = key[0]
            stat[key[0]] = key[1]

        self._total_matches += 1
        # Bankroll, stake, odd plot statistic
        self._stat_df = self._stat_df.append(stat, ignore_index=True)
        # Remove the pending stake from the list
        self._pending_stakes = self._pending_stakes[self._pending_stakes.index != index]

    def skip_match(self):
        self._total_matches += 1

    def __repr__(self):
        return "ROI: %s \n Bankroll: %s \n Hit Rate: %s \n Won Matches: %s \n Played Matches: %s \n Total Matches: %s \n" % (
            self.roi, self.bankroll, self.hitrate, self._matches_won, self._matches_played, self._total_matches)

def full_time_result_eval(self, row, prediction):
    home_goal = row[self._get_col('home goal')]
    away_goal = row[self._get_col('away goal')]

    if home_goal > away_goal and prediction in fulltime_result_tags['Home']:
        return True
    elif away_goal > home_goal and prediction in fulltime_result_tags['Away']:
        return True
    elif away_goal == home_goal and prediction in fulltime_result_tags['Draw']:
        return True
    else:
        return False

def multi_model_decision(self, row):
    # Get all column which contain prediction and probability
    prediction_cols = row[[s for s in row.keys() if self._get_col('prediction') in s]].values.tolist()
    probability_cols = row[[s for s in row.keys() if self._get_col('probability') in s]].values.tolist()
    # Get all model decisions
    model_decisions = np.array(list(zip(prediction_cols, probability_cols)))

    # True False decision
    true_false = np.array(list(map(self.b_model.prediction_check, model_decisions)))

    # Get the betting model decisions about the prediction
    supervised_results = model_decisions[true_false == True]

    if len(supervised_results) > 0:
        # Get the probability scalars only and cast it to float
        num_only = np.asarray(supervised_results[:, 1], dtype=np.float64)

        # Select the maximum from the probabilities
        decision = np.argmax(supervised_results, axis=0)

        # The 1st element will contain the index with the corresponding result
        return supervised_results[decision[1]]
    else:
        return (prediction_cols[0], probability_cols[0])

class BackTestEngine():

    default_config = {
        'stake_limits': np.array([1, np.inf]),
        'init_bankroll': 100,
        'column_names': {
            'date' : 'Date',
            'home': 'Home',
            'away': 'Away',
            'prediction': 'Prediction',
            'probability': 'Probability',
            'home goal': 'Home Goal',
            'away goal': 'Away Goal',
            'home odds': 'Home Odds',
            'away odds': 'Away Odds',
            'draw odds': 'Draw Odds',
        },
        'use_group': False,
        'grouping': {
            'groupby': 'date',
            'window': '1D'
        },
        'use_sort': False,
        'sort': {
            'sortby': ['date'],
            'ascending': [True]
        },
        'use_date_index': False,
        'date_format': "%Y-%m-%d",
        'eval': full_time_result_eval,
        'statistic': StatisticModel,
        'track_unplayed_matches': False,
        'multi_model_decision': multi_model_decision
    }

    def __init__(self,name=DEFAULT_ENGINE_NAME,sig=DEFAULT_ENGINE_NAME, ver=DEFAULT_ENGINE_VERSION, *args, **kwargs):
        # Common parameters
        self._name = name
        self._ver = ver
        self._sig = sig
        # Can be loaded later.
        self._nn_models = listify(kwargs.get('nn_models', None))
        self._b_model = kwargs.get('base_models', None)

        # Store the configuration
        self._config = { **BackTestEngine.default_config , **kwargs.get('config', {})}

    @property
    def b_model(self):
        return self._b_model

    @property
    def stat(self):
        return self._stats

    def _get_stat_index(self, row):
        date_key = None
        if self._config['use_date_index']:
            try:
                date_key = (self._get_col('date'), row[self._get_col('date')])
            except Exception:
                pass
        return date_key

    def simulate(self, dataframe=None):
        assert isinstance(dataframe, pd.DataFrame), "dataframe input parameters must be a pandas DataFrame object."
        assert dataframe is not None or hasattr(self, '_data'), "No input data provided to the engine."
        assert hasattr(self._config['eval'], '__call__'), "Match evaulation config parameter must be a callable."
        self.reset()
        dataframe = self._data if dataframe is None else dataframe
        # Calculate the predictions
        prediction_df = self._predict(dataframe)
        # Concat the prediction and input data
        merged_df = pd.concat([dataframe.reset_index(drop=True), prediction_df.reset_index(drop=True)], axis=1)
        # Prepare the input data
        merged_df = self._prepare_data(merged_df)
        # If no grouping was selected
        if type(merged_df) == pd.DataFrame:
            for idx, row in merged_df.iterrows():
                try:
                    history = merged_df[(merged_df.index < idx)]
                    # Calculate the stake, odd and result
                    stake, odd, result = self._calculate_stake(row, history)
                    # If the stake > 0 this mean we really want to bet on this match.
                    if stake > 0.0 or self._config['track_unplayed_matches']:
                        # Limit the stake according to the configuration
                        stake = max(min(stake, self._config['stake_limits'][1]), self._config['stake_limits'][0])
                        # Place the bet with the stake
                        idx = self._stats.place_bet(stake, odd, result)
                        # Get the statistic index
                        date_key = self._get_stat_index(row)
                        # Evaluate the bet
                        self._stats.eval_bet(idx, date_key)
                    else:
                        self._stats.skip_match()
                    # Yield the statistic data
                    yield self.stat

                except AssertionError as err:
                    logger.error(err)
                    break
        else:
            for name, group in merged_df:
                idx_list = []
                key_row = None
                for idx, row in group.iterrows():
                    key_row = row
                    # TODO: Fixme, histroy is needed
                    #history = merged_df[(merged_df.index < idx)]
                    history = None
                    # Calculate the stake, odd and result
                    stake, odd, result = self._calculate_stake(row, history)
                    # If the stake > 0 this mean we really want to bet on this match.
                    if stake > 0.0 or self._config['track_unplayed_matches']:
                        # Limit the stake according to the configuration
                        stake = max(min(stake, self._config['stake_limits'][1]), self._config['stake_limits'][0])
                        # Place the bet with the stake
                        idx_list.append( self._stats.place_bet(stake, odd, result) )
                    else:
                        self._stats.skip_match()

                # Get the statistic index
                date_key = self._get_stat_index(key_row)
                # Evaluate all the pending bets
                [self._stats.eval_bet(idx, date_key) for idx in idx_list]
                # Yield the statistic data
                yield self.stat

    def run_simulate(self, dataframe=None):
        assert isinstance(dataframe, pd.DataFrame), "dataframe input parameters must be a pandas DataFrame object."
        for _ in self.simulate(dataframe):
            pass
        return self._stats


    def reset(self):
        self._stats = self._config['statistic'](self._config['init_bankroll'])


    def load_models(self, nn_models, betting_model):
        from frameworks.base_models.nn_model import IModel
        from frameworks.base_models.b_model import IBmodel

        for nn_model in listify(nn_models):
            assert isinstance(nn_model, IModel), "Neural network model has to be an instance of IModel"
        assert isinstance(betting_model, IBmodel), "Betting model has to be an instance of IBmodel"
        self._nn_models = listify(nn_models)
        self._b_model = betting_model

    def load_data(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame), "dataframe input parameters must be a pandas DataFrame object."
        self._data = dataframe

    def _calculate_stake(self, row, history=None):
        # Get the supervised decision
        prediction, probability = self._config['multi_model_decision'](self, row)

        # Get all the rows which contains odds
        odds_row = row[[s for s in row.keys() if 'Odds' in s]]
        # Get the correct odds for the prediction
        odd = self._get_prediction_odds((prediction, probability), odds_row)

        # Evaluate the result
        won = self._config['eval'](self, row, prediction)

        bankroll = self._stats.bankroll
        stake = self._b_model.calculate(bankroll=bankroll, prediction=prediction, probability=probability, odd=odd, history=history, statistic=self._stats)
        # Return with all the calcualted data
        return stake, odd, won

    def _get_prediction_odds(self, prediction_tuple, row):
        prediction, _ = prediction_tuple
        if prediction == 'H' or prediction == 'Home':
            return row[self._get_col('home odds')]
        elif prediction == 'A' or prediction == 'Away':
            return row[self._get_col('away odds')]
        elif prediction == 'D' or prediction == 'Draw':
            return row[self._get_col('draw odds')]
        else:
            logger.error("Prediction is not Home/Away/Draw and currently is not supported.")
            return None

    def _get_col(self, col_sig):
        return self._config['column_names'][col_sig.lower()]

    def _check_input_data(self, dataframe):
        cols = self._config['column_names'].values()
        assert cols in dataframe.columns, "Mandatory columns %s are not present in the input dataframe." % cols

    def _predict(self, dataframe):
        # Check if the engine configuration is okay
        self._check_engine()
        # Do the prediction
        predict_col_name = "%s " + self._get_col('prediction')
        proba_col_name = "%s " + self._get_col('probability')
        prediction_array = []
        for nn_model in self._nn_models:
            prediction = nn_model.predict(dataframe)
            df = pd.DataFrame(prediction)

            #TODO: Because of reasons?? The property method 'name' is not working here
            name1 = predict_col_name % nn_model._name
            name2 = proba_col_name % nn_model._name
            # Rename the columns
            df.columns = [name1, name2 ]
            prediction_array.append(df)

        return pd.concat(prediction_array, axis=1).reset_index(drop=True)


    def _prepare_data(self, dataframe):
        date_col = self._get_col('date')
        if ('sort' in self._config and self._config['use_sort']) or ('grouping' in self._config and self._config['use_group']):
            dataframe[date_col] = pd.to_datetime(dataframe[date_col], format=self._config['date_format'])
        if 'sort' in self._config and self._config['use_sort']:
            cols = [self._get_col(sig) for sig in self._config['sort']['sortby']]
            order = self._config['sort']['ascending']
            dataframe = dataframe.sort_values(cols, ascending=order).reset_index(drop=True)
        if 'grouping' in self._config and self._config['use_group']:
            if self._config['grouping']['groupby'] == 'date':
                dataframe = dataframe.groupby(pd.Grouper(key=date_col, freq=self._config['grouping']['window']))
            else:
                cols = [self._get_col(sig) for sig in self._config['grouping']['groupby']]
                dataframe = dataframe.groupby(cols)
        return dataframe


    def _check_engine(self):
        assert self._nn_models[0] is not None, "At least one neural network model."
        assert self._b_model is not None, "One betting model has to be initialized."
        assert hasattr(self, '_stats'), "statistic object is not added to the model"