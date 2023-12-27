class get_path:
    """
    Các function này dùng để load các đường dẫn thư mục sử dụng.
    data_sensor là load đường dẫn chứa các file csv đã đo được từ cảm biến
    data_csv là file đã tổng hợp dữ liệu
    calib_csv là file chứ dữ liệu calib
    save_csv là tổng hơp các đường dẫn lưu file
    """

    @staticmethod
    def data_sensor(name_folder):
        path = fr'data\data_sensor\{name_folder}'
        return path

    @staticmethod
    def data_cvs(name_file):
        path = fr'data\final_data\{name_file}.csv'
        return path

    @staticmethod
    def calib_cvs(name_file):
        path = fr'data\calib_data\{name_file}.csv'
        return path

    @staticmethod
    def save_csv():
        train_test_path = r'data\train_test data'
        return train_test_path
