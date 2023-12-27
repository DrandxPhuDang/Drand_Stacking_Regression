from kennard_stone import train_test_split
from helper.drand_path_data import get_path
import pandas as pd
import numpy as np


class split_data:
    @staticmethod
    def split(X_data, y_data, test_size, features, target):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size)

        all_data = pd.concat([pd.DataFrame(np.array(y_data), columns=[target]),
                              pd.DataFrame(np.array(X_data), columns=features)], axis=1)
        train_all = pd.concat([pd.DataFrame(np.array(y_train), columns=[target]),
                               pd.DataFrame(np.array(X_train), columns=features)], axis=1)
        test_all = pd.concat([pd.DataFrame(np.array(y_test), columns=[target]),
                              pd.DataFrame(np.array(X_test), columns=features)], axis=1)

        train_test_path = get_path.save_csv()
        all_data.to_csv(fr'{train_test_path}\all.csv', index=False, header=True, na_rep='Unknown')
        train_all.to_csv(fr'{train_test_path}\train.csv', index=False, header=True, na_rep='Unknown')
        test_all.to_csv(fr'{train_test_path}\test.csv', index=False, header=True, na_rep='Unknown')

        return X_train, X_test, y_train, y_test
