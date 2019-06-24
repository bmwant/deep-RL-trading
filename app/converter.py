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
    db_size = 119  # 100 steps + 20 steps for testings (windows = 10)
    for i in range(3):
        db_data = rounded[i*db_size:(i+1)*db_size]
        db_name = 'db{:02d}.csv'.format(i)
        db_path = os.path.join(db_dir, db_name)
        np.savetxt(db_path, db_data, fmt='%d', delimiter=',')


if __name__ == '__main__':
    main()
