import pandas as pd
from helper.drand_grid_search import Gridsearch_svr, Gridsearch_etr, Gridsearch_r, Gridsearch_pls, Gridsearch_xgb


class get_models:
    @staticmethod
    def grid_search_model(X_train, y_train):
        """
        Function này dùng để tìm các tham số tốt nhất cho mô hình.
        best_model_svr, best_model_etr, best_model_r, best_model_pls, best_model_xgb các biến này là mô hình tốt nhất
        khi đã tìm được tham số tốt nhất.
        :param X_train:
        :param y_train:
        :return:
        """
        global best_model_svr, best_model_etr, best_model_r, best_model_pls, best_model_xgb
        best_model_svr = Gridsearch_svr(X_train, y_train)
        best_model_etr = Gridsearch_etr(X_train, y_train)
        best_model_r = Gridsearch_r(X_train, y_train)
        best_model_pls = Gridsearch_pls(X_train, y_train,
                                        features=pd.DataFrame(X_train).iloc[0, 0:])
        best_model_xgb = Gridsearch_xgb(X_train, y_train)

    @staticmethod
    def base_models():
        """
        Model con trong base model chỉ chỉnh sửa khi hiểu các bản chất của từng mô hình. Base_models này gần như đã ổn
        định và không cần thay đổi

        :return:
        """
        base_models = [
            ('etr', best_model_etr.best_estimator_),
            ('svr', best_model_svr.best_estimator_),
            ('xgb', best_model_xgb.best_estimator_),
            ('pls', best_model_pls.best_estimator_),
        ]
        return base_models

    @staticmethod
    def meta_model():
        """
        Meta_model là mô hình có khả năng chống over fitting nên cũng không cần sửa. Nếu chưa tìm
        được mô hình nào tốt hơn
        :return:
        """
        meta_model = best_model_r.best_estimator_
        return meta_model
