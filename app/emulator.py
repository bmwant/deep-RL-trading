from typing import List
from abc import ABC, abstractmethod
from collections import deque

import click
import numpy as np

from app import lib
from app.visualizer import show_state


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


class Transaction(object):
    def __init__(self, buy: float, sale: float):
        self.buy = buy
        self.sale = sale


class Environment(object):
    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_valid_actions(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_reward(self, *args, **kwargs):
        pass


class Market(Environment):
    """
    state
        MA of prices, normalized using values at t
        ndarray of shape (window_state, n_instruments * n_MA), i.e., 2D
        which is self.state_shape

    action
        three actions
        0:	sell
        1:	buy
        2: 	hold/idle
    """
    def __init__(
        self,
        sampler,
        window_state,
        open_cost,
        direction=1.,
        max_transactions: int = lib.MAX_TRANSACTIONS,
    ):
        # where to get data from
        self.sampler = sampler
        # how many days (records) of history we can lookup
        self.window_state = window_state
        # price for position opening
        self.open_cost = open_cost

        # just default values
        self.direction = direction  # 1.

        # number of possible actions
        self.n_action = 3
        # shape of a state
        # e.g. (40, 1) for univariate and (40, 2) for bivariate
        self.state_shape = (window_state, self.sampler.n_var)
        # labels for actions
        self.action_labels = [
            'sell',
            'buy',
            'idle',
        ]
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
        self._last_price = 0.
        self._last_price_norm = 0.

        self.max_transactions = max_transactions
        self.transactions = deque(maxlen=max_transactions)

    def reset(self, rand_price=True):
        self.empty = True
        self.transactions = deque(maxlen=self.max_transactions)
        if rand_price:
            prices, self.title = self.sampler.sample()
            # get only first signal
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

    def get_valid_actions(self) -> List[int]:
        """
        0 - sell stock share/usd
        1 - buy stock share/usd
        2 - hold stock share/idle action
        """
        if len(self.transactions) == self.max_transactions:
            return [0, 2]  # sell, idle
        elif len(self.transactions) == 0:
            return [1, 2]  # buy, idle
        else:
            return [0, 1, 2]  # sell, buy, idle

    def get_noncash_reward(self, t=None, empty=None):
        if t is None:
            t = self.t
        if empty is None:
            empty = self.empty
        # reward = self.direction * (self.prices_norm[t+1] - self.prices_norm[t])
        # when selling our profit is a diff
        # we have to use normalized prices to be able properly leverage
        # replay buffer
        reward = self.prices_norm[t+1] - self.prices_norm[t]
        if empty:
            # when buying price is higher, so we loose some money
            reward -= self.open_cost
        # if reward < 0:
        #     reward *= (1. + self.risk_averse)
        return reward

    def step(self, action):
        done = False
        if action == 0:  # wait/close
            reward = 0.
            self.empty = True
        elif action == 1:  # open
            reward = self.get_noncash_reward()
            # reward = self.get_reward(action)
            self.empty = False
        elif action == 2:  # keep prev state; idle; do nothing
            reward = self.get_noncash_reward()
            # reward = self.get_reward(action)
            # empty value is the same as previously
        else:
            raise ValueError('No such action: %s' % action)

        self.t += 1
        state = self.get_state()
        # show_state(self.prices, state)
        return (
            state,  # state
            reward,  # reward
            self.t == self.t_max,  # done?
            self.get_valid_actions()  # allowed actions
        )

    def stepv1(self, action, verbose=False):
        verbose = False
        if action == 0:  # sell
            t = self.transactions.popleft()
            price = self.prices[self.t][0]  # price now
            # t.sale - price that was previously
            diff = price - t.sale  # our profit
            reward = price
            if verbose:
                click.secho('\n+', fg='green', nl=False)
                print('%.2f->%.2f; %.2f' %
                      (t.sale, price, diff))
        elif action == 1:  # we buy usd
            # add to list of transactions
            t = Transaction(
                buy=self.prices[self.t][0],
                sale=self.prices[self.t][1],
            )
            self.transactions.append(t)
            reward = -t.sale
            if verbose:
                click.secho('-', fg='red', nl=False)
        elif action == 2:  # hold/do nothing
            reward = -0.2  # do not encorage waiting?
            if verbose:
                click.secho('_', fg='blue', nl=False)
        else:
            raise ValueError('No such action: %s' % action)

        self.t += 1
        done = self.t == self.t_max
        state = self.get_state()
        # show_state(self.prices, state)
        return (
            state,  # state
            reward,  # reward
            done,
            self.get_valid_actions()  # allowed actions
        )

    def get_reward(self, action, t=None):
        if t is None:
            t = self.t
        reward = self.prices[t+1] - self.prices[t]
        # print('{} <-{}-> {}'.format(self.prices[self.t-1], self.prices[self.t], self.prices[self.t+1]))
        if action == 0:
            reward = 0
        elif action == 1:  # buy usd
            reward -= self.open_cost
        elif action == 2:  # sell usd
            pass

        return reward

    def step_verbose(self, action):
        print('\nTimestep: {}'.format(self.t))
        # print('Price is: {}'.format(self.prices[self.t]))
        if action == 0:  # do nothing
            reward = 0
            self.empty = True  # empty transaction
            print('Agent decided to idle')
        elif action == 1:  # buy usd
            reward = self.get_reward(action)
            self.empty = False
            print('Agent decided to buy')
        elif action == 2:  # sell money, profit is a diff
            reward = self.get_reward(action)
            print('Agent decided to sell')
        else:
            raise ValueError('No such action: %s' % action)

        self.t += 1
        state = self.get_state()
        done = self.t == self.t_max
        actions = self.get_valid_actions()
        print('Done?', done)
        print('Reward is', reward)
        print('Available actions', actions)
        # show_state(self.prices, state)
        return (
            state,
            reward,
            done,
            actions,
        )

    @property
    def hanging(self):
        return sum([t.buy for t in self.transactions])


class PlayTransaction(object):
    def __init__(self, price: float):
        self.price = price


class PlayMarket(Environment):
    def __init__(self, sampler, window_state: int):
        self.sampler = sampler
        self.window_state = window_state

        self.max_slots = lib.MAX_TRANSACTIONS
        self.transactions = deque(maxlen=self.max_slots)

        self.title: str = ''
        self.prices = []

        # todo (misha): remove me
        self.max_profit = 1

        # self.state_shape = (window_state, self.sampler.n_var+1)
        self.state_shape = (window_state + self.max_slots, 1)
        # labels for actions
        self.action_labels = [
            'sell',
            'buy',
            'idle',
        ]
        self.n_action = len(self.action_labels)  # 3 available actions

        self.t = None
        self.t0 = window_state - 1
        self.t_max = None

    def reset(self, rand_price=True, training=True):
        self.transactions = deque(maxlen=self.max_slots)
        if rand_price:
            self.prices, self.title = self.sampler.sample(training)
            # self.prices = np.reshape(prices[:, 0], prices.shape[0]).copy()

        self.t = self.t0
        self.t_max = len(self.prices) - 1

        # assert we have required data points
        assert self.t_max - self.t + self.window_state == \
            self.sampler.EPISODE_LENGTH
        # todo (misha): maybe set state shape here?
        return self.get_state(), self.get_valid_actions()

    def get_state(self, *args, **kwargs):
        start_i = self.t - self.window_state + 1
        end_i = self.t + 1
        state = self.prices[start_i:end_i].copy()
        # (can't buy, can't sale) pair
        # state = np.append(state, [
        #     len(self.transactions) == self.max_slots,
        #     len(self.transactions) == 0,
        # ])
        slots = np.zeros((self.window_state, 1), dtype=np.float32)
        for i, t in enumerate(self.transactions):
            slots[i] = t.price

        # return np.hstack((state, slots)).copy()
        state = np.append(state, slots)
        return np.expand_dims(state, axis=1)

    def get_valid_actions(self):
        """
        0 - sell stock share/usd
        1 - buy stock share/usd
        2 - hold stock share/idle action
        """
        if len(self.transactions) == self.max_slots:
            return [0, 2]  # sell, idle
        elif len(self.transactions) == 0:
            return [1, 2]  # buy, idle
        else:
            return [0, 1, 2]  # sell, buy, idle

    def step(self, action, verbose=True):
        if action == 0:  # sell
            slot = self.transactions.popleft()
            price = self.prices[self.t][0]
            diff = price - slot.price  # profit value
            reward = price
        elif action == 1:  # buy
            slot = PlayTransaction(price=self.prices[self.t][0])
            self.transactions.append(slot)
            reward = -slot.price
        elif action == 2:  # idle
            reward = -0.2
        else:
            raise ValueError('No such action: %s' % action)

        self.t += 1
        done = self.t == self.t_max
        state = self.get_state()
        valid_actions = self.get_valid_actions()
        return (
            state,
            reward,
            done,
            valid_actions,
        )

    def get_reward(self, *args, **kwargs):
        pass


def test_play_environment():
    from app.sampler import PlaySampler
    from app.plots import plot_state_prices_window

    sampler = PlaySampler(db_name='db2018_train.csv')
    env = PlayMarket(
        sampler=sampler,
        window_state=10,
    )

    state, actions = env.reset(rand_price=True)
    price_window = state.transpose()[0]
    plot_state_prices_window(price_window)


if __name__ == '__main__':
    test_play_environment()
