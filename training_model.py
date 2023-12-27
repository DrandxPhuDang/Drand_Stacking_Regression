from helper.drand_stacking_model import stacking_regression
from helper.drand_models import get_models
from helper.drand_print_score import print_score, cal_rpd
from helper.drand_split_data import split_data
from helper.drand_get_data import get_data


class training:
    def __init__(self, path_data_all, start_col, test_size, target):
        super().__init__()
        """
        X là các cột giá trị tại bước sóng từ 900 - 1367 nm
        y là biến mục tiêu (target) thường là Brix, Acid, DM, ...
        features là tên của các bước sóng từ 900 - 1367 nm
        """
        X, y, features = get_data.get_X_y(path_data_all, start_col=start_col)

        """
        Chia dữ liệu bằng kennard stone
        X_train, y_train dùng để huấn luyện mô hình hồi quy
        X_test, y_test dùng để kiểm tra mô hình hồi quy
        """
        X_train, X_test, y_train, y_test = split_data.split(X, y, test_size=test_size, features=features, target=target)

        """
        Tìm các tham số tốt nhất cho mô hình, dòng code khởi tạo nhập vào X và y là bộ dữ liệu cần tìm tham số tốt nhất
        X, y thường là X_train, y_train
        """
        get_models.grid_search_model(X_train, y_train)

        """
        Tạo model gồm base_models và meta_model
        base_models là tập hợp các model con dùng để training từ dữ liệu phổ đưa ra giá trị làm đầu vào cho meta_model
        meta_model là model dùng để đưa ra dự đoán cuối cùng
        """
        base_models = get_models.base_models()
        meta_model = get_models.meta_model()

        """
        Dòng code stacking_regression(base_models, meta_model) khởi tạo model training
        Dòng code model.fit(X_train, y_train) bắt đầu huấn luyện mô hình
        """
        model = stacking_regression(base_models, meta_model)
        model.fit(X_train, y_train)

        """
        Dự đoán kết quả
        """
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        """
        In ra điểm đánh giá mô hình
        Nếu R và R^2 đều hơn 0.8 ở tệp train lẫn tệp test thì mô hình được đánh giá là tốt
        Hoặc xem xét thêm RPD nếu lớn hơn bằng 2 thì mô hình tốt
        """
        print("----------- Train -----------------")
        print_score(y_train, y_pred_train)
        print("------------ Test -----------------")
        print_score(y_test, y_pred_test)
        cal_rpd(y_test, y_pred_test)

