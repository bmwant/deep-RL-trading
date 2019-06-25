import os
import numpy as np

import click

from app.lib import makedirs
from app.lib import DEFAULT_MA_WINDOW
from app.visualizer import show_step


class Simulator(object):
    def __init__(
        self, agent, env, visualizer, fld_save,
        ma_window: int = DEFAULT_MA_WINDOW,
    ):
        self.agent = agent
        self.env = env
        # range for moving average window
        self.ma_window = ma_window
        self.visualizer = visualizer
        self.fld_save = fld_save  # directory where to save results
        self._best_result = 0

    def play_one_episode(
        self,
        exploration,
        training=True,
        rand_price=True,
        verbose=False,
    ):
        state, valid_actions = self.env.reset(
            rand_price=rand_price, training=training)
        done = False
        env_t = 0
        try:
            env_t = self.env.t
        except AttributeError:
            pass

        cum_rewards = [np.nan] * env_t
        actions = [np.nan] * env_t  # history of previous actions
        states = [None] * env_t  # history of previous states
        prev_cum_rewards = 0.

        while not done:
            action = self.agent.act(state, exploration, valid_actions)
            next_state, reward, done, valid_actions = self.env.step(
                action, verbose=verbose,)
            # next_state, reward, done, valid_actions = self.env.step_verbose(action)

            cum_rewards.append(prev_cum_rewards+reward)
            # import ipdb; ipdb.set_trace()
            # print('Step', self.env.t)
            # print('Reward', reward)
            # print(cum_rewards)
            # print('='*30)
            # print()
            prev_cum_rewards = cum_rewards[-1]
            actions.append(action)
            states.append(next_state)

            if training:
                self.agent.remember(state, action, reward, next_state, done, valid_actions)
                self.agent.replay()

            state = next_state
        return cum_rewards, actions, states

    def train(
        self,
        n_episode,
        *,
        save_per_episode=10,
        exploration_init=1.,
        exploration_decay=0.995,
        exploration_min=0.01,
        verbose=True,
    ):
        fld_model = os.path.join(self.fld_save, 'model')
        makedirs(fld_model)	 # don't overwrite if already exists
        with open(os.path.join(fld_model, 'QModel.txt'), 'w') as f:
            f.write(self.agent.model.qmodel)

        exploration = exploration_init
        fld_save = os.path.join(self.fld_save, 'training')
        makedirs(fld_save)

        safe_total_rewards = []
        explored_total_rewards = []
        explorations = []  # store to visualize later
        path_record = os.path.join(fld_save, 'record.csv')

        with open(path_record, 'w') as f:
            f.write('episode,game,exploration,explored_reward,'
                    'safe_reward,MA_explored,MA_safe\n')

        for n in range(n_episode):
            print('{}/{} training...'.format(n, n_episode))
            exploration = max(exploration_min, exploration * exploration_decay)
            explorations.append(exploration)
            explored_cum_rewards, explored_actions, _ = self.play_one_episode(
                exploration,
                rand_price=True,  # use new data for each new episode
                verbose=True,
            )
            explored_total_rewards.append(explored_cum_rewards[-1])

            # Safe values: exploration is completely disabled
            safe_cum_rewards, safe_actions, _ = self.play_one_episode(
                exploration=0,  # exploit existing model
                training=False,  # do not append to replay buffer
                rand_price=False,  # reuse previous sampled prices
            )
            safe_total_rewards.append(safe_cum_rewards[-1])

            MA_total_rewards = np.median(
                explored_total_rewards[-self.ma_window:])
            MA_safe_total_rewards = np.median(
                safe_total_rewards[-self.ma_window:])

            ss = [
                str(n),
                self.env.title.replace(',', ';'),
                '%.1f' % (exploration*100.),  # exploration factor
                '%.1f' % (explored_total_rewards[-1]),  # explored rewards
                '%.1f' % (safe_total_rewards[-1]),  # safe rewards
                '%.1f' % MA_total_rewards,  # MA explored rewards
                '%.1f' % MA_safe_total_rewards,  # MA safe rewards
            ]

            with open(path_record, 'a') as f:
                f.write(','.join(ss)+'\n')

            last_reward = safe_cum_rewards[-1]
            profit = last_reward
            if verbose:
                # print('Hanging', self.env.hanging)
                header = [
                    '#',
                    'Data used',
                    'Exploration, %',
                    '[E] reward',
                    '[S] reward',
                    'MA [E] reward',
                    'MA [S] reward',
                ]
                explored_rewards = '%.2f' % (explored_cum_rewards[-1])
                safe_rewards = '%.2f' % (safe_cum_rewards[-1])
                if explored_cum_rewards[-1] > 0:
                    explored_rewards = click.style(explored_rewards, fg='green')
                if safe_cum_rewards[-1] > 0:
                    safe_rewards = click.style(safe_rewards, fg='green')

                data = [[
                    n,  # current episode
                    self.env.title,  # data label used for episode
                    '%.1f' % (exploration * 100.),
                    explored_rewards,
                    safe_rewards,
                    '%.2f' % MA_total_rewards,
                    '%.2f' % MA_safe_total_rewards,
                    # '%.2f' % profit,
                ]]
                show_step(data=data, header=header)

            # if n % save_per_episode == 0:
            if last_reward > self._best_result:
                print('{} saving results...'.format(n))
                self.agent.save(fld_model)
                self._best_result = last_reward

                """
                self.visualizer.plot_a_episode(
                    self.env, self.agent.model, 
                    explored_cum_rewards, explored_actions,
                    safe_cum_rewards, safe_actions,
                    os.path.join(fld_save, 'episode_%i.png'%(n)))
                """

        if self.visualizer is not None:
            print('Plotting episodes', fld_save)
            self.visualizer.plot_episodes(
                explored_total_rewards,
                safe_total_rewards,
                explorations,
                os.path.join(fld_save, 'total_rewards.png'),
            )

    def test(
        self, n_episode, *, save_per_episode=10, subfld='testing', verbose=True,
    ):
        """
        Test on `n_episode` episodes, disable exploration, use only trained
        model.
        """
        fld_save = os.path.join(self.fld_save, subfld)
        makedirs(fld_save)
        safe_total_rewards = []
        path_record = os.path.join(fld_save, 'record.csv')

        with open(path_record, 'w') as f:
            f.write('episode,game,safe_reward,MA_safe\n')

        for n in range(n_episode):
            print('{}/{} testing...'.format(n, n_episode))

            safe_cum_rewards, safe_actions, _ = self.play_one_episode(
                0,
                training=False,
                rand_price=True,
                verbose=verbose,
            )

            last_reward = safe_cum_rewards[-1]
            safe_total_rewards.append(last_reward)
            MA_safe_total_rewards = np.median(
                safe_cum_rewards[-self.ma_window:])
            ss = [
                str(n),  # number of episode
                self.env.title.replace(',', ';'),
                '%.1f' % (safe_cum_rewards[-1]),  # pnl, safe cumulative rewards
                '%.1f' % MA_safe_total_rewards  # moving average on safe total rewards
            ]

            with open(path_record, 'a') as f:
                f.write(','.join(ss)+'\n')

            if verbose:
                header = [
                    '# (testing)',
                    'Data used',
                    '[S] reward',
                    'MA [S] reward',
                ]

                safe_reward = '%.2f' % last_reward
                if last_reward > 0:
                    safe_reward = click.style(safe_reward, fg='green')

                data = [[
                    n,  # current episode
                    self.env.title,  # data label used for episode
                    safe_reward,
                    '%.2f' % MA_safe_total_rewards,
                ]]
                print()
                show_step(data=data, header=header)

            if n % save_per_episode == 0:
                from app.plots import show_actions
                show_actions(safe_actions)

            if self.visualizer is not None:
                self.visualizer.plot_a_episode(
                    self.env,
                    self.agent.model,
                    [np.nan]*len(safe_cum_rewards),
                    [np.nan]*len(safe_actions),
                    safe_cum_rewards,
                    safe_actions,
                    os.path.join(fld_save, 'episode_%i.png' % n)
                )
                """
                self.visualizer.plot_episodes(
                    None, safe_total_rewards, None, 
                    os.path.join(fld_save, 'total_rewards.png'),
                    MA_window)
                """


if __name__ == '__main__':
    a = [1, 2, 3]
    print(np.mean(a[-100:]))
