import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def print_score(y_actual, y_predicted):
    """
    :param y_actual: Nhập y_data từ dãy X_data đã dự đoán
    :param y_predicted: Nhập y_pred
    :return: Trả về 4 đánh giá (R, R Square, R_MSE, MAE)
    example: '''Accuracy score'''
            print('--------------- TRAIN--------------------')
            print_score(self.y_train, y_pred_train)
            print('--------------- TEST--------------------')
            score_test = print_score(self.y_test, y_pred_test)
            Để lấy giá trị R, R_Squared, R_MSE, MAE thì chỉ cần thêm Score_test[0] (0-3)
            tương ứng thứ tự của các giá trị
    """
    R = np.corrcoef(y_actual, y_predicted, rowvar=False)
    print('R:', "{:.3f}".format(R[0][1]))
    R_Squared = r2_score(y_actual, y_predicted)
    print('R^2:', "{:.3f}".format(R_Squared))
    R_MSE = math.sqrt(mean_squared_error(y_actual, y_predicted))
    print('R_MSE :', "{:.3f}".format(R_MSE))
    MAE = mean_absolute_error(y_actual, y_predicted)
    print('MAE:', "{:.3f}".format(MAE))
    return R, R_Squared, R_MSE, MAE


def cal_rpd(actual_values, predictions):
    """
    :param actual_values: Nhập vào giá trị y thực tế
    :param predictions: Nhập vào giá trị y dự đoán
    :return: Trả về kết quả RPD
    example: print('--------------- RPD--------------------')
            RPD_Test = cal_rpd(self.y_test, y_pred_test)
            print('RPD:', "{:.2f}".format(RPD_Test))
    """
    actual_values = np.ravel(actual_values)
    predictions = np.ravel(predictions)
    sd_actual = np.std(actual_values)
    error = (predictions - actual_values)
    bias = np.mean(predictions - actual_values)
    sep = np.sqrt(np.mean((error - bias) ** 2))
    rpd = sd_actual / sep
    print('RPD:', "{:.3f}".format(rpd))
    return rpd
