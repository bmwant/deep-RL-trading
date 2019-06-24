import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

from app.lib import DATA_DIR


def plot_data(data):
    x_ticks = np.arange(0, len(data) + 1, step=1)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    y_ticks = np.arange(1, 11, step=1)
    # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    plt.bar(np.arange(len(data)), data, align='edge', width=1., color='tab:pink')
    plt.ylabel('Normalized price')
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.title('Converted rates data')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
