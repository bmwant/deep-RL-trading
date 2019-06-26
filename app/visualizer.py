from typing import List

import click
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from terminaltables.other_tables import SingleTable

from app.lib import DEFAULT_MA_WINDOW


def get_tick_labels(bins, ticks):
    ticklabels = []
    for i in ticks:
        if i < len(bins):
            ticklabels.append('%.2f' % (bins[int(i)]))
        else:
            ticklabels.append('%.2f' % (bins[-1])+'+')

    return ticklabels


def show_state(prices, state, offset: int = 0):
    plt.rcParams['figure.figsize'] = (10, 6)
    colors = ['tab:blue', 'tab:orange']

    prices_ticks = np.arange(0, len(prices))
    ax_top = plt.subplot(211)
    ax_top.set_title('Current dataset: {}'.format(len(prices)))
    ax_top.set_prop_cycle('color', colors)
    plt.plot(prices_ticks, prices)
    # Build vertical lines to show current state window
    plt.axvline(x=offset, color='m', linestyle=':')
    plt.axvline(x=len(state)+offset, color='m', linestyle=':')
    # c - cyan
    # m - magenta
    state_ticks = np.arange(offset, len(state)+offset)
    ax_bottom = plt.subplot(212)
    ax_bottom.set_title('Current state windows: {}'.format(len(state)))
    ax_bottom.set_prop_cycle('color', colors)
    plt.plot(state_ticks, state)

    # This should be called after all axes have been added
    plt.tight_layout()  # minimize overlaps
    plt.show()


class Visualizer(object):
    def __init__(self, action_labels, ma_window: int = DEFAULT_MA_WINDOW):
        self.n_action = len(action_labels)
        self.action_labels = action_labels
        self.ma_window = ma_window

    def plot_a_episode(
        self,
        env,
        model,
        explored_cum_rewards,
        explored_actions,
        safe_cum_rewards,
        safe_actions,
        fig_path,
    ):
        f, axs = plt.subplots(3, 1, sharex=True, figsize=(14, 14))
        ax_price, ax_action, ax_Q = axs

        ls = ['-', '--']
        for i in range(min(2, env.prices.shape[1])):
            p = env.prices[:, i]/env.prices[0, i]*100 - 100
            ax_price.plot(p, 'k'+ls[i], label='input%i - 100' % i)

        ax_price.plot(explored_cum_rewards, 'b', label='explored P&L')
        ax_price.plot(safe_cum_rewards, 'r', label='safe P&L')
        ax_price.legend(loc='best', frameon=False)
        ax_price.set_title(
            '%s, safe: %.1f, explored: %1.f' %
            (env.title, safe_cum_rewards[-1], explored_cum_rewards[-1]))

        ax_action.plot(explored_actions, 'b', label='explored')
        ax_action.plot(safe_actions, 'r', label='safe', linewidth=2)
        ax_action.set_ylim(-0.4, self.n_action-0.6)
        ax_action.set_ylabel('action')
        ax_action.set_yticks(range(self.n_action))
        ax_action.legend(loc='best', frameon=False)

        style = ['k', 'r', 'b']
        qq = []
        for t in range(env.t0):
            qq.append([np.nan] * self.n_action)
        for t in range(env.t0, env.t_max):
            qq.append(model.predict(env.get_state(t)))
        for i in range(self.n_action):
            ax_Q.plot([float(qq[t][i]) for t in range(len(qq))],
                style[i], label=self.action_labels[i])
        ax_Q.set_ylabel('Q')
        ax_Q.legend(loc='best', frameon=False)
        ax_Q.set_xlabel('t')

        plt.subplots_adjust(wspace=0.4)
        plt.savefig(fig_path)
        plt.close()

    def plot_episodes(
        self,
        explored_total_rewards,
        safe_total_rewards,
        explorations,
        fig_path,
    ):
        f = plt.figure(figsize=(14, 10))  # width, height in inch (100 pixel)
        if explored_total_rewards is None:
            f, ax_reward = plt.subplots()
        else:
            figshape = (3, 1)
            ax_reward = plt.subplot2grid(figshape, (0, 0), rowspan=2)
            ax_exploration = plt.subplot2grid(figshape, (2, 0), sharex=ax_reward)

        tt = range(len(safe_total_rewards))

        if explored_total_rewards is not None:
            ma = pd.Series(np.array(explored_total_rewards)) \
                .rolling(window=self.ma_window, min_periods=1).median()
            std = pd.Series(np.array(explored_total_rewards)) \
                .rolling(window=self.ma_window, min_periods=3).std()
            ax_reward.plot(tt, explored_total_rewards, 'bv', fillstyle='none')
            ax_reward.plot(tt, ma, 'b', label='explored ma', linewidth=2)
            ax_reward.plot(tt, std, 'b--', label='explored std', linewidth=2)

        ma = pd.Series(np.array(safe_total_rewards)) \
            .rolling(window=self.ma_window, min_periods=1).median()
        std = pd.Series(np.array(safe_total_rewards)) \
            .rolling(window=self.ma_window, min_periods=3).std()
        ax_reward.plot(tt, safe_total_rewards, 'ro', fillstyle='none')
        ax_reward.plot(tt, ma, 'r', label='safe ma', linewidth=2)
        ax_reward.plot(tt, std, 'r--', label='safe std', linewidth=2)

        ax_reward.axhline(y=0, color='k', linestyle=':')
        ax_reward.set_ylabel('total reward')
        ax_reward.legend(loc='best', frameon=False)
        ax_reward.yaxis.tick_right()
        ylim = ax_reward.get_ylim()
        ax_reward.set_ylim(
            min(min(explored_total_rewards), min(safe_total_rewards)),
            max(max(explored_total_rewards), max(safe_total_rewards)),
        )

        if explored_total_rewards is not None:
            ax_exploration.plot(tt, np.array(explorations)*100., 'k')
            ax_exploration.set_ylabel('exploration')
            ax_exploration.set_xlabel('episode')

        plt.savefig(fig_path)
        plt.close()


class VisualizerSequential(object):

    def config(self):
        pass

    def __init__(self, model):
        self.model = model
        self.layers = []
        for layer in self.model.layers:
            self.layers.append(str(layer.name))

        self.inter_models = dict()
        model_input = self.model.input
        for layer in self.layers:
            self.inter_models[layer] = keras.models.Model(
                                inputs=model_input,
                                outputs=self.model.get_layer(layer).output)
        self.config()


class VisualizerConv1D(VisualizerSequential):

    def config(self):
        self.n_channel = self.model.input.shape[2]
        n_col = self.n_channel
        for layer in self.layers:
            shape = self.inter_models[layer].output.shape
            if len(shape) == 3:
                n_col = max(n_col, shape[2])

        self.figshape = (len(self.layers)+1, int(n_col))

    def plot(self, x):

        f = plt.figure(figsize=(30,30))

        for i in range(self.n_channel):
            ax = plt.subplot2grid(self.figshape, (0,i))
            ax.plot(x[0,:,i], '.-')
            ax.set_title('input, channel %i' % i)

        for i_layer in range(len(self.layers)):
            layer = self.layers[i_layer]
            z = self.inter_models[layer].predict(x)
            print('plotting '+layer)
            if len(z.shape) == 3:
                for i in range(z.shape[2]):
                    ax = plt.subplot2grid(self.figshape, (i_layer+1, i))
                    ax.plot(z[0,:,i], '.-')
                    ax.set_title(layer+' filter %i'%i)
            else:
                ax = plt.subplot2grid(self.figshape, (i_layer+1, 0))
                ax.plot(z[0,:], '.-')
                ax.set_title(layer)

        ax.set_ylim(-100, 100)

    def print_w(self):
        layer = self.layers[0]
        ww = self.inter_models[layer].get_weights()
        for w in ww:
            print(w.shape)
            print(w)


def show_step(data: List, header=None):
    table_data = []
    if header is not None:
        if len(data[0]) != len(header):
            raise ValueError(
                'Size of header should be equal to size of data row')
        table_data.append(header)
    table_data.extend(data)
    tbl = SingleTable(table_data)
    click.echo(tbl.table)


def test_visualizer():
    f = plt.figure()  # figsize=(5,8))
    axs_action = []
    ncol = 3
    nrow = 2

    clim = (0, 1)

    ax = plt.subplot2grid((nrow, ncol), (0, ncol - 1))
    ax.matshow(np.random.random((2, 2)), cmap='RdYlBu_r', clim=clim)

    for action in range(3):
        row = 1 + action / ncol
        col = action % ncol
        ax = plt.subplot2grid((nrow, ncol), (row, col))
        cax = ax.matshow(np.random.random((2, 2)), cmap='RdYlBu_r', clim=clim)

    ax = plt.subplot2grid((nrow, ncol), (0, 0), colspan=ncol-1)
    cbar = f.colorbar(cax, ax=ax)

    plt.show()


if __name__ == '__main__':
    test_visualizer()
