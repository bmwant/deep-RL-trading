import numpy as np


def find_ideal(p, just_once=False):
    """
    Find all possible successful price differences either for one day step
    for the whole dataset or just one biggest difference across the data.
    """
    if not just_once:
        diff = np.array(p[1:]) - np.array(p[:-1])
        return sum(np.maximum(np.zeros(diff.shape), diff))
    else:
        best = 0.
        for i in range(len(p)-1):
            best = max(best, max(p[i+1:]) - p[i])

        return best


class Market(object):
    """
    state
        MA of prices, normalized using values at t
        ndarray of shape (window_state, n_instruments * n_MA), i.e., 2D
        which is self.state_shape

    action
        three actions
        0:	empty, don't open/close.
        1:	open a position
        2: 	keep a position
    """
    def __init__(
        self,
        sampler,
        window_state,
        open_cost,
        direction=1.,
        risk_averse=0.
    ):
        # where to get data from
        self.sampler = sampler
        # how many days (records) of history we can lookup
        self.window_state = window_state
        # price for position opening
        self.open_cost = open_cost

        # just default values
        self.direction = direction  # 1.
        self.risk_averse = risk_averse  # 0.

        # number of possible actions
        self.n_action = 3
        # shape of a state
        # e.g. (40, 1) for univariate and (40, 2) for bivariate
        self.state_shape = (window_state, self.sampler.n_var)
        # labels for actions
        self.action_labels = ['empty', 'open', 'keep']
        # initial time step
        # we need `window_state` days to look back, so starting here
        self.t0 = window_state - 1
        self.t = None  # current time step
        self.t_max = None  # maximum possible time step
        self.empty = True

        # Initialization fields
        self.max_profit = 0
        self.title = ''
        self.prices = []  # ndarray for the data from sampler
        self.prices_norm = []  # ndarray for normalized price dataset

    def reset(self, rand_price=True):
        self.empty = True
        if rand_price:
            prices, self.title = self.sampler.sample()
            price = np.reshape(prices[:, 0], prices.shape[0])

            self.prices = prices.copy()
            self.prices_norm = price / price[0] * 100
            self.t_max = len(self.prices_norm) - 1

        # assuming we open positions on each time they will be successful
        self.max_profit = find_ideal(self.prices_norm[self.t0:])
        # starting time step equals to `t0` to have `window_state`
        # history lookup
        self.t = self.t0
        return self.get_state(), self.get_valid_actions()

    def get_state(self, t=None):
        """
        Get normalized values of prices for the given `window_state` days range
        """
        if t is None:
            t = self.t
        state = self.prices[t - self.window_state + 1: t + 1, :].copy()
        for i in range(self.sampler.n_var):
            norm = np.mean(state[:, i])
            # todo: not sure why do we need exactly this normalization
            state[:, i] = (state[:, i]/norm - 1.)*100
        return state

    def get_valid_actions(self):
        if self.empty:
            return [0, 1]  # wait, open
        else:
            return [0, 2]  # close, keep

    def get_noncash_reward(self, t=None, empty=None):
        if t is None:
            t = self.t
        if empty is None:
            empty = self.empty
        reward = self.direction * (self.prices_norm[t+1] - self.prices_norm[t])
        if empty:
            reward -= self.open_cost
        if reward < 0:
            reward *= (1. + self.risk_averse)
        return reward

    def step(self, action):
        done = False
        if action == 0:  # wait/close
            reward = 0.
            self.empty = True
        elif action == 1:  # open
            reward = self.get_noncash_reward()
            self.empty = False
        elif action == 2:  # keep
            reward = self.get_noncash_reward()
        else:
            raise ValueError('No such action: %s' % action)

        self.t += 1
        return (
            self.get_state(),  # state
            reward,  # reward
            self.t == self.t_max,  # done?
            self.get_valid_actions()  # allowed actions
        )
