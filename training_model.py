import pandas as pd
from kennard_stone import train_test_split
from helper.drand_stacking_model import stacking_regression
from helper.drand_models import get_models
from helper.drand_print_score import print_score, cal_rpd
from helper.drand_split_data import split_data
from helper.drand_get_data import get_data


class training:
    def __init__(self, path_data_all, start_col=12, test_size=0.2):
        super().__init__()
        # Lấy dữ liệu
        X, y, features = get_data.get_X_y(path_data_all, start_col=start_col)

        # Chia dữ liệu train_test
        X_train, X_test, y_train, y_test = split_data.split(X, y, test_size=test_size, features=features)

        # Chọn model cho huấn luyện mô hình hồi quy xếp chồng
        base_models = get_models.base_models()
        meta_model = get_models.meta_model()

        # Huấn luyện mô hình hồi quy xếp chồng
        model = stacking_regression(base_models, meta_model)
        model.fit(X_train, y_train)

        # Dự đoán kết quả
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        print("----------- Train -----------------")
        print_score(y_train, y_pred_train)
        print("------------ Test -----------------")
        print_score(y_test, y_pred_test)
        cal_rpd(y_test, y_pred_test)

