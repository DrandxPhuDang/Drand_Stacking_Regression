import pandas as pd


class get_data:
    @staticmethod
    def get_X_y(path_data_all, start_col=12):
        data_all = pd.read_csv(path_data_all)
        list_features = data_all.iloc[:0, start_col:]
        features = [f'{e}' for e in list_features]
        X = data_all[features]
        y = data_all['Brix']
        return X, y, features
