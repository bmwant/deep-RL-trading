import os
import json
import random
import pickle
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from app.lib import ROOT_DIR, DEFAULT_DATASET_LENGTH


class Sampler(object):
    def __init__(self):
        self.db = None
        self.i_db = 0
        self.n_db = None
        self.sample = None
        self.n_var = 0
        self.title = ''
        self.attrs = []

    def load_db(self, fld):
        self.i_db = 0
        self.db = pickle.load(open(os.path.join(fld, 'db.pickle'), 'rb'))
        param = json.load(open(os.path.join(fld, 'param.json'), 'rb'))
        self.n_db = param['n_episodes']
        self.sample = self.__sample_db
        for attr in param:
            if hasattr(self, attr):
                setattr(self, attr, param[attr])
        self.title = 'DB_'+param['title']

    def build_db(self, n_episodes, fld):
        db = []
        for i in range(n_episodes):
            prices, title = self.sample()
            db.append((prices, '[{}]_{}'.format(i, title)))
        os.makedirs(fld)  # don't overwrite existing fld
        pickle.dump(db, open(os.path.join(fld, 'db.pickle'), 'wb'))
        param = {'n_episodes': n_episodes}
        for k in self.attrs:
            param[k] = getattr(self, k)
        json.dump(param, open(os.path.join(fld, 'param.json'), 'w'))

    def __sample_db(self):
        prices, title = self.db[self.i_db]
        self.i_db += 1
        if self.i_db == self.n_db:
            self.i_db = 0
        return prices, title


class PBSampler(Sampler):
    def __init__(self):
        super().__init__()
        self.offset = 0
        self.n_var = 2  # price only
        fld = os.path.join(ROOT_DIR, 'data', 'PBSamplerDB')
        self.load_db(fld)

    def load_db(self, fld):
        db_path = os.path.join(fld, 'uah_to_usd_2018.csv')
        df = pd.read_csv(db_path)
        self.db = df[['buy', 'sale']].to_numpy()
        self.sample = self.__sample_db
        # number of episodes equals to number of samples available
        self.n_db = self.db.shape[0] - DEFAULT_DATASET_LENGTH + 1

    def __sample_db(self) -> Tuple[np.ndarray, str]:
        s = self.db[self.offset:DEFAULT_DATASET_LENGTH+self.offset]
        self.title = 'uah_to_usd_2018_{}'.format(self.offset)
        self.offset += 1
        if self.offset == self.n_db:
            warnings.warn(
                'Last sample, will be reusing data starting next invocation')
            self.offset = 0
        return s, self.title

    def __len__(self):
        """
        Number of samples available
        """
        return self.n_db


class PlaySampler(Sampler):
    def __init__(
        self,
        db_name: str,
        episode_length: int = DEFAULT_DATASET_LENGTH,
        testing: bool = False,
    ):
        super().__init__()
        self.db_name = db_name
        self.n_var = 1
        db_path = os.path.join(ROOT_DIR, 'data', 'PlaySamplerDB', db_name)
        self.load_db(db_path=db_path)
        # number of samples
        self.episode_length = episode_length
        self.n_db = self.db.shape[0] - self.episode_length + 1
        self.testing = testing

    def reset(self):
        self.i_db = 0

    def load_db(self, db_path):
        self.db = np.genfromtxt(db_path, delimiter=',')
        # self.db = np.expand_dims(db, axis=1)
        self.sample = self.__sample_db

    def __sample_db(self, training: bool = True) -> Tuple[np.ndarray, str]:
        s = self.db[self.i_db:self.i_db+self.episode_length, :]
        self.i_db += 1
        self.title = '{}_{}'.format(self.db_name, self.i_db)

        # cycle data
        if self.i_db == self.n_db:
            self.i_db = 0

        return s, self.title

    @property
    def test_samples(self):
        return self.n_db


class PairSampler(Sampler):
    def __init__(
        self,
        game,
        window_episode=None,
        forecast_horizon_range=None,
        max_change_perc=10.,
        noise_level=10.,
        n_section=1,
        fld=None,
        windows_transform=None,
    ):
        super().__init__()
        self.window_episode = window_episode
        self.forecast_horizon_range = forecast_horizon_range
        self.max_change_perc = max_change_perc
        self.noise_level = noise_level
        self.n_section = n_section
        self.windows_transform = windows_transform or []
        self.n_var = 2 + len(self.windows_transform)  # price, signal

        self.attrs = [
            'title', 'window_episode', 'forecast_horizon_range',
            'max_change_perc', 'noise_level', 'n_section', 'n_var'
        ]
        param_str = str((self.noise_level, self.forecast_horizon_range, self.n_section, self.windows_transform))

        if game == 'load':
            self.load_db(fld)
        elif game in ['randwalk', 'randjump']:
            self.__rand = getattr(self, '_PairSampler__'+game)
            self.sample = self.__sample
            self.title = game + param_str
        else:
            raise ValueError

    def __randwalk(self, l):
        change = (np.random.random(l + self.forecast_horizon_range[1]) - 0.5) * 2 * self.max_change_perc/100
        forecast_horizon = random.randrange(self.forecast_horizon_range[0], self.forecast_horizon_range[1])
        return change[:l], change[forecast_horizon: forecast_horizon + l], forecast_horizon

    def __randjump(self, l):
        change = [0.] * (l + self.forecast_horizon_range[1])
        n_jump = random.randrange(15, 30)
        for i in range(n_jump):
            t = random.randrange(len(change))
            change[t] = (np.random.random() - 0.5) * 2 * self.max_change_perc/100
        forecast_horizon = random.randrange(self.forecast_horizon_range[0], self.forecast_horizon_range[1])
        return change[:l], change[forecast_horizon: forecast_horizon + l], forecast_horizon

    def __sample(self):

        L = self.window_episode
        if bool(self.windows_transform):
            L += max(self.windows_transform)
        l0 = L/self.n_section
        l1 = L

        d_price = []
        d_signal = []
        forecast_horizon = []

        for i in range(self.n_section):
            if i == self.n_section - 1:
                l = l1
            else:
                l = l0
                l1 -= l0
            d_price_i, d_signal_i, horizon_i = self.__rand(l)
            d_price = np.append(d_price, d_price_i)
            d_signal = np.append(d_signal, d_signal_i)
            forecast_horizon.append(horizon_i)

        price = 100. * (1. + np.cumsum(d_price))
        signal = 100. * (1. + np.cumsum(d_signal)) + \
                np.random.random(len(price)) * self.noise_level

        price += (100 - min(price))
        signal += (100 - min(signal))

        inputs = [price[-self.window_episode:], signal[-self.window_episode:]]
        for w in self.windows_transform:
            inputs.append(signal[-self.window_episode - w: -w])

        return np.array(inputs).T, 'forecast_horizon='+str(forecast_horizon)


class SinSampler(Sampler):
    def __init__(
        self,
        game,
        window_episode=None,
        noise_amplitude_ratio=None,
        period_range=None,
        amplitude_range=None,
        fld=None,
    ):
        super().__init__()
        self.n_var = 1  # price only

        self.window_episode = window_episode
        self.noise_amplitude_ratio = noise_amplitude_ratio
        self.period_range = period_range
        self.amplitude_range = amplitude_range
        self.can_half_period = False

        self.attrs = [
            'title',
            'window_episode',
            'noise_amplitude_ratio',
            'period_range',
            'amplitude_range',
            'can_half_period',
        ]

        param_str = str((
            self.noise_amplitude_ratio, self.period_range, self.amplitude_range
        ))
        if game == 'single':
            self.sample = self.__sample_single_sin
            self.title = 'SingleSin'+param_str
        elif game == 'concat':
            self.sample = self.__sample_concat_sin
            self.title = 'ConcatSin'+param_str
        elif game == 'concat_half':
            self.can_half_period = True
            self.sample = self.__sample_concat_sin
            self.title = 'ConcatHalfSin'+param_str
        elif game == 'concat_half_base':
            self.can_half_period = True
            self.sample = self.__sample_concat_sin_w_base
            self.title = 'ConcatHalfSin+Base'+param_str
            self.base_period_range = (int(2*self.period_range[1]), 4*self.period_range[1])
            self.base_amplitude_range = (20, 80)
        elif game == 'load':
            self.load_db(fld)
        else:
            raise ValueError

    def __rand_sin(
        self,
        period_range=None,
        amplitude_range=None,
        noise_amplitude_ratio=None,
        full_episode=False,
    ):
        if period_range is None:
            period_range = self.period_range
        if amplitude_range is None:
            amplitude_range = self.amplitude_range
        if noise_amplitude_ratio is None:
            noise_amplitude_ratio = self.noise_amplitude_ratio

        period = random.randrange(period_range[0], period_range[1])
        amplitude = random.randrange(amplitude_range[0], amplitude_range[1])
        noise = noise_amplitude_ratio * amplitude

        if full_episode:
            length = self.window_episode
        else:
            if self.can_half_period:
                length = int(random.randrange(1,4) * 0.5 * period)
            else:
                length = period

        p = 100. + amplitude * np.sin(np.array(range(length)) * 2 * 3.1416 / period)
        p += np.random.random(p.shape) * noise

        return p, '100+%isin((2pi/%i)t)+%ie' % (amplitude, period, noise)

    def __sample_concat_sin(self):
        prices = []
        p = []
        while True:
            p = np.append(p, self.__rand_sin(full_episode=False)[0])
            if len(p) > self.window_episode:
                break
        prices.append(p[:self.window_episode])
        return np.array(prices).T, 'concat sin'

    def __sample_concat_sin_w_base(self):
        prices = []
        p = []
        while True:
            p = np.append(p, self.__rand_sin(full_episode=False)[0])
            if len(p) > self.window_episode:
                break
        base, base_title = self.__rand_sin(
            period_range=self.base_period_range,
            amplitude_range=self.base_amplitude_range,
            noise_amplitude_ratio=0.,
            full_episode=True)
        prices.append(p[:self.window_episode] + base)
        return np.array(prices).T, 'concat sin + base: '+base_title

    def __sample_single_sin(self):
        prices = []
        funcs = []
        p, func = self.__rand_sin(full_episode=True)
        prices.append(p)
        funcs.append(func)
        return np.array(prices).T, str(funcs)


def test_sin_sampler():
    window_episode = 180
    window_state = 40
    noise_amplitude_ratio = 0.5
    period_range = (10, 40)
    amplitude_range = (5, 80)
    game = 'concat_half_base'
    instruments = ['fake']

    sampler = SinSampler(
        game,
        window_episode,
        noise_amplitude_ratio,
        period_range,
        amplitude_range,
    )
    n_episodes = 100
    """
    for i in range(100):
        plt.plot(sampler.sample(instruments)[0])
        plt.show()
        """
    fld = os.path.join(
        ROOT_DIR,
        'data',
        'SinSamplerDB',
        game+'_B',
    )
    sampler.build_db(n_episodes, fld)


def test_pair_sampler():
    fhr = (10, 30)  # forecast horizon range
    n_section = 1
    max_change_perc = 30.  # max change from previous value (in percentage)
    noise_level = 5
    game = 'randjump'
    window_episode = 180
    windows_transform = []  # no transformations

    sampler = PairSampler(
        game,
        window_episode=window_episode,  # length for the data generated
        forecast_horizon_range=fhr,
        n_section=n_section,
        noise_level=noise_level,
        max_change_perc=max_change_perc,
        windows_transform=windows_transform,
    )

    # plt.plot(sampler.sample()[0])
    # plt.show()

    game_name = '{game}_{window_episode},{n_section}{fhr}{transforms}_{suffix}'.format(
        game=game,
        window_episode=window_episode,
        n_section=n_section,
        fhr=fhr,
        transforms=windows_transform,
        suffix='B',
    )
    # output directory
    fld = os.path.join(
        ROOT_DIR,
        'data',
        'PairSamplerDBTest',
        game_name,
    )
    print('Generating data:')
    print('\tNumber of episodes: {}'.format(window_episode))
    print('\tNumber of sections?: {}'.format(n_section))
    print('\tForecast horizon: {}'.format(fhr))
    print('\tWindows transform?: {}'.format(windows_transform))
    print('Writing data to directory:\n{}'.format(fld))
    sampler.build_db(window_episode, fld)


def test_pb_sampler():
    from app.visualizer import show_state

    sampler = PBSampler()
    print('Number of samples available', len(sampler))
    prices, title = sampler.sample()
    price = np.reshape(prices[:, 0], prices.shape[0])
    print(price, price.shape)
    state = prices[:30]  # one month slice
    show_state(prices, state)


def test_play_sampler():
    from app.visualizer import show_state

    sampler = PlaySampler('uah_to_usd_2017_both_scaled_1_10.csv')
    prices, title = sampler.sample()
    print(prices, prices.shape)
    state = prices[:10]  # ten data points
    show_state(prices, state)


if __name__ == '__main__':
    # test_sin_sampler()
    # test_pair_sampler()
    # test_pb_sampler()
    test_play_sampler()
