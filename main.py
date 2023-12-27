from training_model import training
from helper.drand_path_data import get_path

"""
get_path.data_cvs(name_file='Data A') là dòng code load file data tổng hợp
path_calib_data = get_path.data_cvs(name_file='final_data_calibration (25-10-2023)') là dòng code load file data calib
"""
path_data_all = get_path.data_cvs(name_file='Data A')
path_calib_data = get_path.data_cvs(name_file='final_data_calibration (25-10-2023)')


"""
Chạy chương trình huấn luyện
Lưu ý: Running chương trình tại file main.py
"""
training(path_data_all, start_col=12, test_size=0.2, target='Brix')
