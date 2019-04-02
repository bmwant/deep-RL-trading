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


class Market:
    """
    state MA of prices, normalized using values at t
        ndarray of shape (window_state, n_instruments * n_MA), i.e., 2D
					which is self.state_shape

	action 			three action
					0:	empty, don't open/close. 
					1:	open a position
					2: 	keep a position
    """

    def reset(self, rand_price=True):
        self.empty = True
        if rand_price:
            prices, self.title = self.sampler.sample()
            price = np.reshape(prices[:, 0], prices.shape[0])

            self.prices = prices.copy()
            self.price = price / price[0] * 100
            self.t_max = len(self.price) - 1
        print(self.price)
        self.max_profit = find_ideal(self.price[self.t0 :], False)
        self.t = self.t0
        self._step = 0
        return self.get_state(), self.get_valid_actions()

    def get_state(self, t=None):
        """
        State is a window of `window_state` days back
        """
        if t is None:
            t = self.t
        i = self._step
        state = self.prices[t - self.window_state + 1 : t + 1, :].copy()
        # import pdb; pdb.set_trace()
        print('shape', state.shape)
        for i in range(self.sampler.n_var):
            norm = np.mean(state[:, i])
            state[:, i] = (state[:, i] / norm - 1.0) * 100
        return state

    def get_valid_actions(self):
        if self.amount:
            return [0, 1]  # we can sale or idle
        else:
            return [0, 2]  # we can buy or idle

    def get_noncash_reward(self, t=None, empty=None):
        if t is None:
            t = self.t
        if empty is None:
            empty = self.empty
        reward = self.direction * (self.price[t + 1] - self.price[t])
        if empty:
            reward -= self.open_cost
        if reward < 0:
            reward *= 1.0 + self.risk_averse
        return reward

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
        done = self._step == self.size
        return (
            self.get_state(),
            reward,
            done,
            self.get_valid_actions(),
        )

    def __init__(
        self, sampler, window_state, open_cost, direction=1.0, risk_averse=0.0
    ):

        self.sampler = sampler
        self.window_state = window_state
        self.open_cost = open_cost
        self.direction = direction
        self.risk_averse = risk_averse

        self.n_action = 3
        self.state_shape = (window_state, self.sampler.n_var)
        self.action_labels = ['idle', 'sell', 'buy']
        self.t0 = window_state - 1
        self._df = load_year_dataframe(2018)
        self._df['date'] = pd.to_datetime(self._df['date'], format=DATE_FMT)

        self._df.sort_values(by=['date'], inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        self._df = self._df.head(180)
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
        sampler=sampler,
        window_state=31,
        open_cost=0.0,
    )
    print(env.size)
    print(env._df)

    state, valid_actions = env.reset(rand_price=True)
    print('State', state)
    print('Valid actions', valid_actions)
