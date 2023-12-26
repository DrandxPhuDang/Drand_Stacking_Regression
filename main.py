from training_model import training
from helper.drand_path_data import get_path
from export_data import Export_Data

# Các đường dẫn file dữ liệu
path_data_all = get_path.data_cvs()
path_calib_data = get_path.data_cvs()

# Huấn luyện mô hình hồi quy
training(path_data_all)
