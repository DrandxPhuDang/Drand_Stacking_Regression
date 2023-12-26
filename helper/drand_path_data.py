import pandas as pd
import numpy as np


class get_path:

    @staticmethod
    def data_sensor():
        path = r'D:\Quyt stacking regression\data\data_sensor'
        return path

    @staticmethod
    def data_cvs():
        path = r'D:\Quyt stacking regression\data\final_data\Data A.csv'
        return path

    @staticmethod
    def calib_cvs():
        path = r'D:\Quyt stacking regression\data\calib_data\final_data_calibration (25-10-2023).csv'
        return path

    @staticmethod
    def save_csv():
        train_test_path = r'D:\Quyt stacking regression\data\train_test data'
        file_data_export = r'D:\Quyt stacking regression\data\demo_data'
        return train_test_path, file_data_export
