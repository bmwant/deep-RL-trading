import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from app.lib import DATA_DIR


def main():
    db_path = os.path.join(DATA_DIR, 'PBSamplerDB', 'uah_to_usd_2018.csv')
    df = pd.read_csv(db_path)

    rate = df[['nb']].to_numpy(dtype=np.float32)
    scaled_rate = minmax_scale(rate, feature_range=(1, 10))
    rounded = scaled_rate.round()

    db_dir = os.path.join(DATA_DIR, 'PlaySamplerDB')
    db_name = 'db2018_train.csv'
    db_path = os.path.join(db_dir, db_name)
    np.savetxt(db_path, rounded, fmt='%d', delimiter=',')


if __name__ == '__main__':
    main()
