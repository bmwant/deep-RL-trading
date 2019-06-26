import os

import click
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


@click.command()
@click.argument('in_file', type=click.File('r'))
@click.argument('out_file', type=click.File('w'))
def convert(in_file, out_file):
    click.secho('Reading data from %s...' % in_file.name, fg='blue')
    df = pd.read_csv(in_file)

    rate = df[['buy', 'sale']].to_numpy(dtype=np.float32)
    # both columns to single feature
    rates_feature = np.hstack((rate[:, 0], rate[:, 1]))
    scaled_feature = minmax_scale(rates_feature, feature_range=(1, 10))

    buy_scaled, sale_scaled = np.array_split(scaled_feature, 2)
    buy_scaled_col = np.expand_dims(buy_scaled, axis=1)
    sale_scaled_col = np.expand_dims(sale_scaled, axis=1)
    scaled_rates = np.hstack((buy_scaled_col, sale_scaled_col))
    click.secho('Writing scaled data to %s...' % out_file.name, fg='yellow')
    np.savetxt(out_file, scaled_rates, fmt='%.4f', delimiter=',')
    click.secho('Done.', fg='green')


if __name__ == '__main__':
    convert()
