import os
import numpy as np

import click

from app.lib import makedirs
from app.lib import DEFAULT_MA_WINDOW
from app.plots import show_step_chart, show_episode_chart, show_episodes_chart
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
        self._best_result = float('-inf')

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
        extra = {}  # extra data used for charts
        while not done:
            action = self.agent.act(state, exploration, valid_actions)
            next_state, reward, done, valid_actions = self.env.step(
                action,
                verbose=verbose,
            )
            # next_state, reward, done, valid_actions = self.env.step_verbose(action)

            cum_rewards.append(prev_cum_rewards+reward)
            prev_cum_rewards = cum_rewards[-1]
            actions.append(action)
            states.append(next_state)

            if training:
                self.agent.remember(state, action, reward, next_state, done, valid_actions)
                self.agent.replay()

            state = next_state
            if verbose and not training:
                steps_path = os.path.join(self.fld_save, 'steps')
                makedirs(steps_path)
                save_path = os.path.join(
                    steps_path, 'step_{:03d}'.format(self.env.t))
                show_step_chart(
                    prices=self.env.prices,
                    slots=self.env.slots.transpose(),
                    actions=actions,
                    step=self.env.t,
                    window_state=self.env.window_state,
                    save_path=save_path,
                )
        extra['profit'] = self.env._profit_abs
        return cum_rewards, actions, states, extra

    def train(
        self,
        n_episode,
        *,
        save_per_episode=10,
        exploration_init=1.,
        exploration_decay=0.995,
        exploration_min=0.01,
        verbose=True,
        chart_per_episode=10,
    ):
        fld_model = os.path.join(self.fld_save, 'model')
        makedirs(fld_model)	 # don't overwrite if already exists
        with open(os.path.join(fld_model, 'QModel.txt'), 'w') as f:
            f.write(self.agent.model.qmodel)

        exploration = exploration_init
        fld_save = os.path.join(self.fld_save, 'training')
        makedirs(fld_save)

        # Store statistics, used for visualization
        safe_total_rewards = []  # for all episodes
        explored_total_rewards = []  # for all episodes
        explorations = []  # for all episodes
        ma_explored_total_rewards = []  # updated after each episode
        ma_safe_total_rewards = []  # updated after each episode
        safe_total_actions = []

        path_record = os.path.join(fld_save, 'record.csv')
        episodes_path = os.path.join(fld_save, 'episodes')
        makedirs(episodes_path)
        with open(path_record, 'w') as f:
            f.write('episode,game,exploration,explored_reward,'
                    'safe_reward,MA_explored,MA_safe\n')

        for n in range(n_episode):
            print('{}/{} training...'.format(n, n_episode))
            extra = {}
            exploration = max(exploration_min, exploration * exploration_decay)
            explorations.append(exploration)
            explored_cum_rewards, explored_actions, _, explored_extra = \
                self.play_one_episode(
                    exploration,
                    rand_price=True,  # use new data for each new episode
                    verbose=True,
            )
            extra['profit_explored'] = explored_extra['profit']
            extra['reward_explored'] = explored_cum_rewards[-1]
            explored_total_rewards.append(explored_cum_rewards[-1])

            # Safe values: exploration is completely disabled
            safe_cum_rewards, safe_actions, _, safe_extra = \
                self.play_one_episode(
                    exploration=0,  # exploit existing model
                    training=False,  # do not append to replay buffer
                    rand_price=False,  # reuse previous sampled prices
            )
            extra['profit_safe'] = safe_extra['profit']
            extra['reward_safe'] = safe_cum_rewards[-1]
            safe_total_rewards.append(safe_cum_rewards[-1])
            safe_total_actions.extend(safe_actions)

            # for all episodes
            ma_explored_total_reward = np.median(
                explored_total_rewards[-self.ma_window:])
            ma_explored_total_rewards.append(ma_explored_total_reward)
            # for all episodes
            ma_safe_total_reward = np.median(
                safe_total_rewards[-self.ma_window:])
            ma_safe_total_rewards.append(ma_safe_total_reward)

            ss = [
                str(n),
                self.env.title.replace(',', ';'),
                '%.1f' % (exploration*100.),  # exploration factor
                '%.1f' % (explored_total_rewards[-1]),  # explored rewards
                '%.1f' % (safe_total_rewards[-1]),  # safe rewards
                '%.1f' % ma_explored_total_reward,  # MA explored rewards
                '%.1f' % ma_safe_total_reward,  # MA safe rewards
            ]

            with open(path_record, 'a') as f:
                f.write(','.join(ss)+'\n')

            last_reward = safe_cum_rewards[-1]
            profit = last_reward
            if verbose:
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
                    '%.2f' % ma_explored_total_reward,
                    '%.2f' % ma_safe_total_reward,
                    # '%.2f' % profit,
                ]]
                show_step(data=data, header=header)

            # if n % save_per_episode == 0:
            if last_reward > self._best_result:
                print('{} saving results...'.format(n))
                self.agent.save(fld_model)
                self._best_result = last_reward

            if n % chart_per_episode == 0:
                save_path = os.path.join(episodes_path, 'episode_{:04d}'.format(n))
                show_episode_chart(
                    episode=n,
                    safe_actions=safe_actions,
                    safe_rewards=safe_cum_rewards,
                    explored_rewards=explored_cum_rewards,
                    exploration=exploration,
                    extra=extra,
                    save_path=save_path,
                )
        save_path = os.path.join(episodes_path, 'summary')
        show_episodes_chart(
            n_episodes=n_episode,
            safe_total_rewards=safe_total_rewards,
            ma_safe_total_rewards=ma_safe_total_rewards,
            explored_total_rewards=explored_total_rewards,
            ma_explored_total_rewards=ma_explored_total_rewards,
            explorations=explorations,
            safe_total_actions=safe_total_actions,
            ma_window=self.ma_window,
            save_path=save_path,
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

            save_all_episodes = False
            if n == 0:
                save_all_episodes = True
            safe_cum_rewards, safe_actions, _, extra = self.play_one_episode(
                0,
                training=False,
                rand_price=True,
                verbose=save_all_episodes,
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
                pass


def linearly_decaying_epsilon(
    decay_period: float,
    step: int,
    warmup_steps: int,
    epsilon_min: float
):
    """
    Returns the current epsilon for the agent's epsilon-greedy policy.
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
    Args:
        decay_period: the period over which epsilon is decayed.
        step: the number of training steps completed so far.
        warmup_steps: the number of steps taken before epsilon is decayed.
        epsilon_min: the final value to which to decay the epsilon parameter.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon_min) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1.0 - epsilon_min)
    return epsilon_min + bonus


if __name__ == '__main__':
    a = [1, 2, 3]
    print(np.mean(a[-100:]))
