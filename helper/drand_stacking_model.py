from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold


class stacking_regression:
    def __init__(self, base_models, meta_model):
        super().__init__()
        global model
        model = StackingRegressor(estimators=base_models, final_estimator=meta_model,
                                  cv=KFold(n_splits=10, shuffle=True, random_state=42), verbose=10)

    @staticmethod
    def fit(X_fit, y_fit):
        model.fit(X_fit, y_fit)

    @staticmethod
    def predict(X_prediction):
        y_prediction = model.predict(X_prediction)
        return y_prediction
