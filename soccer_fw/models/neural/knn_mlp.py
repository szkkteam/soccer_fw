# Common Python library imports

# Pip package imports
from loguru import logger
import pandas as pd
import numpy as np

# Internal package imports
from soccer_fw.core.neural import NeuralEngine
from soccer_fw.utils import fulltime_result_tags


def sum_players_ratings(match_data, players_data='data/ID_mashed.csv'):

    """
    def get_player_rating(ids, date, players_data):
        result_id_filtered = players_data[players_data["SofaID"].isin(ids)]
        if len(result_id_filtered > 0):
            result_date_filtered = result_id_filtered[result_id_filtered["Date"] < date]

            result = result_date_filtered[result_date_filtered['Date'] == result_date_filtered['Date'].max()]

            rating = None
            if result.empty is True:
                result_date_filtered = result_id_filtered[result_id_filtered["Date"] >= date]
                result = result_date_filtered[result_date_filtered['Date'] == result_date_filtered['Date'].min()]
                if result.empty is True:
                    #logger.warn("No rating available for player id: %s" % id)
                    return rating

            rating = result.iloc[0, result.columns.get_loc('Rating')]
            return rating
        else:
            return []
    """

    def get_player_rating(id, date, players_data):
        result_id_filtered = players_data[players_data["SofaID"] == id]
        result_date_filtered = result_id_filtered[result_id_filtered["Date"] < date]
        result = result_date_filtered[result_date_filtered['Date'] == result_date_filtered['Date'].max()]

        if result.empty is True:
            result_date_filtered = result_id_filtered[result_id_filtered["Date"] >= date]
            result = result_date_filtered[result_date_filtered['Date'] == result_date_filtered['Date'].min()]
            # print(result)
            if result.empty is True:
                # id_list = [id]
                # tempdict = collect_unknow_players.scrapping(id_list)
                # rating = int(tempdict["rating_overall"])
                # pre_process.update_rating(self, tempdict)
                # if rating == None:
                rating = None
                return rating

            # self.players_filtered.append(filtered.iloc[0, :])

        rating = result.iloc[0, result.columns.get_loc('Rating')]

        return rating

    if isinstance(match_data, str):
        match_data = pd.read_csv(match_data)
    if isinstance(players_data, str):
        import os
        dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", players_data)
        players_data = pd.read_csv(dir)

    assert isinstance(match_data, pd.DataFrame), "Input object match_data must be a pandas DataFrame object"
    assert isinstance(players_data, pd.DataFrame), "Input object players_data must be a pandas DataFrame object"

    lst = []

    def process(row):
        match_date = row['Date']

        # print(match_date)
        tempdict = {}
        try:
            tempdict['Result'] = row['Result']
        except:
            tempdict['Result'] = False

        for team in ['home', 'away']:
            positions = ['Goalkeeper', 'Defense', 'Midfielder', 'Attacker']
            for p in positions:
                tempdict['%s%s' % (team, p)] = 0

            """
            all_cols = row.keys()
            all_cols = [ elem.replace('_', ' ') for elem in all_cols ]
            id_position = [s for s in all_cols if '%sTeam' % team in s]
            ids = [s for s in id_position if not (("position" in s) or ("Formation" in s))]
            positions = [s for s in id_position if ("position" in s)]
            ids = [ elem.replace(' ', '_') for elem in ids ]
            id_list = row[ids]
            player_ratings = get_player_rating(id_list, match_date, players_data)
            return 1
            """

            for i in range(1, 12):
                player_id = row['%sTeam_%s' % (team, i)]
                player_position = row['%sTeam_%s_position' % (team, i)]

                player_rating = get_player_rating(player_id, match_date, players_data)
                # print(player_rating)
                if player_rating is None:
                    tempdict['%sGoalkeeper' % team] = None
                    tempdict['%sDefense' % team] = None
                    tempdict['%sMidfielder' % team] = None
                    tempdict['%sAttacker' % team] = None
                    break

                if player_position == 'G':
                    tempdict['%sGoalkeeper' % team] += player_rating
                elif player_position == 'D':

                    tempdict['%sDefense' % team] += player_rating
                elif player_position == 'M':
                    tempdict['%sMidfielder' % team] += player_rating
                elif player_position == 'F':
                    tempdict['%sAttacker' % team] += player_rating
        """
        tempdict['MatchId'] = row['MatchId']
        tempdict['HomeTeam'] = row['HomeTeam']
        tempdict['AwayTeam'] = row['AwayTeam']
        tempdict['Home_Odds'] = row['Home_Odds']
        tempdict['Away_Odds'] = row['Away_Odds']
        tempdict['Draw_Odds'] = row['Draw_Odds']
        tempdict['Season'] = row['Season']
        #tempdict['League'] = row['League']
        tempdict['Date'] = match_date
        """

        return row.append(pd.Series(tempdict))

    new_df = match_data.apply(process, axis=1)

        # print(tempdict)
    return new_df


class KNNOnlyDraw(NeuralEngine):

    slug = "knn_only_draw"
    name = "KNN Only Draw"
    version = "v0_1"

    default_params = {
        'n_neighbors': 5,
    }

    features = [
        'homeGoalkeeper', 'homeDefense', 'homeMidfielder', 'homeAttacker',
        'awayGoalkeeper', 'awayDefense', 'awayMidfielder', 'awayAttacker',  # 'Home_Odds', 'Away_Odds'
    ]
    results = 'Result'

    def __init__(self, *args, **kwargs):

        kwargs['params'] = { **KNNOnlyDraw.default_params, **kwargs.get('params',{}) }

        m_kwargs = { **{
            'name': KNNOnlyDraw.name,
            'slug': KNNOnlyDraw.slug,
            'version': KNNOnlyDraw.version
        }, **kwargs}

        super(KNNOnlyDraw, self).__init__(*args, **m_kwargs)

    def init_model(self, model_params):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**model_params)

    def preprocess(self, merged_datasets):
        if isinstance(merged_datasets, list):
            assert len(merged_datasets) == 2, "%s model only accept a list of 2 or 1 input dataset" % self.get_name()
            dataset = sum_players_ratings(merged_datasets[0], merged_datasets[1])
        else:
            dataset = sum_players_ratings(merged_datasets)
        return dataset

    def postprocess(self, validation_data, result_array, **kwargs):
        probability = self._model.predict_proba(validation_data, **kwargs)
        merged_lst = []
        for i in range(0, len(result_array)):
            if result_array[i] == 'H' or  result_array[i] == 'Home':
                merged_lst.append( tuple([result_array[i], probability[i][2]]))
            if result_array[i] == 'A' or  result_array[i] == 'Away':
                merged_lst.append( tuple([result_array[i], probability[i][0]]))
            if result_array[i] == 'D' or result_array[i] == 'Draw':
                merged_lst.append(tuple([result_array[i], probability[i][1]]))
        return np.array(merged_lst)

    def split_dataset(self, merged_dataset, percentage=0.8):
        train_data = merged_dataset[merged_dataset["Season"] != '18/19']
        validation_data = merged_dataset[merged_dataset["Season"] == '18/19']
        return train_data, validation_data

    def select_features(self, x_dataset):
        return x_dataset[KNNOnlyDraw.features], x_dataset[KNNOnlyDraw.results]

    def convert_to_np(self, x_dataset, y_dataset):
        return x_dataset.to_numpy(), y_dataset

    @staticmethod
    def create_model(path_to_file="nn_models/bin", **kwargs):
        model = NeuralEngine.load_from(path_to_file)
        return KNNOnlyDraw(model=model, **kwargs,)

    def help(self):
        return "This model only predict draws."


class MLPCBasic(NeuralEngine):

    slug = "mlpc_basic"
    name = "MLPC Basic Model"
    version = "v0_1"

    default_params = {
        'hidden_layer_sizes': (287,),
        'activation': 'tanh',
        'solver': 'lbfgs',
        'learning_rate': 'adaptive',
        'shuffle': True,
        'random_state': 5
    }

    features = [
        'homeGoalkeeper', 'homeDefense', 'homeMidfielder', 'homeAttacker',
        'awayGoalkeeper', 'awayDefense', 'awayMidfielder', 'awayAttacker',  # 'Home_Odds', 'Away_Odds'
    ]
    results = 'Result'

    def __init__(self, *args, **kwargs):

        kwargs['params'] = { **MLPCBasic.default_params, **kwargs.get('params',{}) }

        m_kwargs = { **{
            'name': MLPCBasic.name,
            'slug': MLPCBasic.slug,
            'version': MLPCBasic.version
        }, **kwargs}

        super(MLPCBasic, self).__init__(*args, **m_kwargs)

    def init_model(self, model_params):
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(**model_params)

    def preprocess(self, merged_datasets):
        if isinstance(merged_datasets, list):
            assert len(merged_datasets) == 2, "%s model only accept a list of 2 or 1 input dataset" % self.get_name()
            dataset = sum_players_ratings(merged_datasets[0], merged_datasets[1])
        else:
            dataset = sum_players_ratings(merged_datasets)
        return dataset

    def split_dataset(self, merged_dataset, percentage=0.8):
        train_data = merged_dataset[merged_dataset["Season"] != '18/19']
        validation_data = merged_dataset[merged_dataset["Season"] == '18/19']
        return train_data, validation_data

    def select_features(self, x_dataset):
        return x_dataset[MLPCBasic.features], x_dataset[MLPCBasic.results]

    def convert_to_np(self, x_dataset, y_dataset):
        return x_dataset.to_numpy(), y_dataset

    def postprocess(self, validation_data, result_array, **kwargs):
        probability = self._model.predict_proba(validation_data, **kwargs)
        merged_lst = []
        for i in range(0, len(result_array)):
            if result_array[i] == 'H' or  result_array[i] == 'Home':
                merged_lst.append( tuple([result_array[i], probability[i][2]]))
            if result_array[i] == 'A' or  result_array[i] == 'Away':
                merged_lst.append( tuple([result_array[i], probability[i][0]]))
            if result_array[i] == 'D' or result_array[i] == 'Draw':
                merged_lst.append(tuple([result_array[i], probability[i][1]]))
        return np.array(merged_lst)

    @staticmethod
    def create_model(path_to_file="nn_models/bin", **kwargs):
        model = NeuralEngine.load_from(path_to_file)
        return MLPCBasic(model=model, **kwargs)

    def help(self):
        return "This model only predict draws."