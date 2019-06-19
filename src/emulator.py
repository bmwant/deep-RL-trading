import sys
sys.path.insert(0, '/Users/bmwant/pr/chemister')

import pandas as pd

from lib import *

from notebooks.helpers import load_year_dataframe, DATE_FMT


def find_ideal(p, just_once):
    if not just_once:
        diff = np.array(p[1:]) - np.array(p[:-1])
        return sum(np.maximum(np.zeros(diff.shape), diff))
    else:
        best = 0.0
        i0_best = None
        for i in range(len(p) - 1):
            best = max(best, max(p[i + 1 :]) - p[i])

        return best


class Market(object):
    """
    state MA of prices, normalized using values at t
        ndarray of shape (window_size, n_instruments * n_MA), i.e., 2D
					which is self.state_shape
    action 	three actions
        0:	empty, don't open/close. 
        1:	open a position
        2: 	keep a position
    """

    def reset(self):
        self.empty = True
        prices = self._df[['buy', 'sale']].to_numpy()
        # if rand_price:
        # prices, self.title = self.sampler.sample()
        price = np.reshape(prices[:, 0], prices.shape[0])
        self.prices = prices.copy()
        self.price = price / price[0] * 100
        self.t_max = len(self.price) - 1
        # print(self.price)
        self.max_profit = find_ideal(self.price[self.t0 :], False)
        self.t = self.t0
        self._step = 0
        self.amount = 0
        return self.get_state(), self.get_valid_actions()

    def get_state(self, t=None):
        """
        State is a window of `window_size` days back
        """
        if t is None:
            t = self.t
        # i = self._step
        state = self.prices[t - self.window_size + 1 : t + 1, :].copy()
        # import pdb; pdb.set_trace()
        # print('shape', state.shape)
        for i in range(self.n_var):
            norm = np.mean(state[:, i])
            state[:, i] = (state[:, i] / norm - 1.0) * 100
        return state

    def get_valid_actions(self):
        if self.amount:
            return [0, 1]  # we can sale or idle
        else:
            return [0, 2]  # we can buy or idle

    def _get_current_rates(self):
        step = self._step
        row = self._df.iloc[[step]]
        return row['buy'].item(), row['sale'].item()

    def step(self, action):
        done = False
        rate_buy, rate_sale = self._get_current_rates()

        if action == 0:  # idle
            reward = 0.0
        elif action == 1:  # sell
            # reward = self.get_noncash_reward()
            reward = 100*rate_buy
            self.amount -= 100
        elif action == 2:  # buy
            # reward = self.get_noncash_reward()
            reward = -100*rate_sale
            self.amount += 100
        else:
            raise ValueError("no such action: %s" % action)

        self.t += 1
        self._step += 1
        done = self.t == self.t_max
        return (
            self.get_state(),
            reward,
            done,
            self.get_valid_actions(),
        )

    def __init__(self, window_size=31, total_days=365):

        # self.sampler = sampler
        self.window_size = window_size
        self.total_days = total_days
        # self.open_cost = open_cost
        # self.direction = direction
        # self.risk_averse = risk_averse

        self.n_action = 3
        self.n_var = 2
        self.title = 'uah-usd-prices'
        self.state_shape = (window_size, self.n_var)
        self.action_labels = ['idle', 'sell', 'buy']
        self.t0 = window_size - 1  # to be able look backwards
        self._df = load_year_dataframe(2018)
        self._df['date'] = pd.to_datetime(self._df['date'], format=DATE_FMT)

        self._df.sort_values(by=['date'], inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        self._df = self._df.head(total_days)
        self._step = 0
        self.amount = 0

    @property
    def size(self):
        return len(self._df.index)


if __name__ == '__main__':
    from sampler import PairSampler

    db = 'randjump_100,1(10, 30)[]_'
    db_type = 'PairSamplerDB'
    fld = os.path.join("..", "data", db_type, db + "A")
    sampler = PairSampler("load", fld=fld)
    env = Market(
        window_size=7,
        total_days=31,
    )
    print(env.size)
    print(env._df)

    state, valid_actions = env.reset()
    print('State', state)
    print('Valid actions', valid_actions)
