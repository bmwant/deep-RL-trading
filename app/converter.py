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

    rate = df[['nb']].to_numpy(dtype=np.float32)
    scaled_rate = minmax_scale(rate, feature_range=(1, 10))

    click.secho('Writing scaled data to %s...' % out_file.name, fg='yellow')
    np.savetxt(out_file, scaled_rate, fmt='%.4f', delimiter=',')
    click.secho('Done.', fg='green')


if __name__ == '__main__':
    convert()
