from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge


class get_models:
    @staticmethod
    def base_models():
        base_models = [
            ('svr', SVR()),
            ('rf', ExtraTreesRegressor(random_state=42)),
            ('xgb', xgb.XGBRegressor()),
            ('pls', PLSRegression())
        ]
        return base_models

    @staticmethod
    def meta_model():
        meta_model = Ridge()
        return meta_model
