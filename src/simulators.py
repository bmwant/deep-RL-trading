from lib import *
from visualizer import Visualizer


class Simulator:
    def play_one_episode(
        self, exploration, training=True, print_t=False
    ):

        state, valid_actions = self.env.reset()
        done = False
        env_t = 0
        try:
            env_t = self.env.t
        except AttributeError:
            pass

        cum_rewards = [np.nan] * env_t
        actions = [np.nan] * env_t
        states = [None] * env_t
        prev_cum_rewards = 0.0

        while not done:
            if print_t:
                print(self.env.t)

            action = self.agent.act(state, exploration, valid_actions)
            next_state, reward, done, valid_actions = self.env.step(action)
            # print('Getting reward', reward)

            cum_rewards.append(prev_cum_rewards + reward)
            prev_cum_rewards = cum_rewards[-1]
            actions.append(action)
            states.append(next_state)

            if training:
                self.agent.remember(
                    state, action, reward, next_state, done, valid_actions
                )
                self.agent.replay()

            state = next_state

        return cum_rewards, actions, states

    def train(
        self,
        n_episode,
        save_per_episode=10,
        exploration_decay=0.995,
        exploration_min=0.01,
        print_t=False,
        exploration_init=1.0,
    ):

        fld_model = os.path.join(self.fld_save, "model")
        makedirs(fld_model)  # don't overwrite if already exists
        with open(os.path.join(fld_model, "QModel.txt"), "w") as f:
            f.write(self.agent.model.qmodel)

        exploration = exploration_init
        fld_save = os.path.join(self.fld_save, "training")

        makedirs(fld_save)
        MA_window = 14  # MA of performance
        safe_total_rewards = []
        explored_total_rewards = []
        explorations = []
        path_record = os.path.join(fld_save, "record.csv")

        with open(path_record, "w") as f:
            f.write(
                "episode,game,exploration,explored,safe,MA_explored,MA_safe\n"
            )
        for n in range(n_episode):

            print('\n[%d] training...' % n)
            exploration = max(exploration_min, exploration * exploration_decay)
            explorations.append(exploration)
            explored_cum_rewards, explored_actions, _ = self.play_one_episode(
                exploration, print_t=print_t
            )
            # print('Env max profit is', self.env.max_profit)
            explored_total_rewards.append(
                explored_cum_rewards[-1]
            )
            # Safe - playing episode without exploration
            safe_cum_rewards, safe_actions, _ = self.play_one_episode(
                0, training=False, print_t=False
            )
            safe_total_rewards.append(
                safe_cum_rewards[-1]
            )

            MA_total_rewards = np.median(explored_total_rewards[-MA_window:])
            MA_safe_total_rewards = np.median(safe_total_rewards[-MA_window:])

            ss = [
                str(n),
                self.env.title.replace(",", ";"),
                "%.1f" % (exploration * 100.0),
                "%.1f" % (explored_total_rewards[-1]),
                "%.1f" % (safe_total_rewards[-1]),
                "%.1f" % MA_total_rewards,
                "%.1f" % MA_safe_total_rewards,
            ]

            with open(path_record, "a") as f:
                f.write(",".join(ss) + "\n")
            # print("\t".join(ss))
            print(
                'exploration: %.1f; '
                'explored reward: %.1f; '
                'safe reward: %.1f' % (
                    exploration*100, 
                    explored_total_rewards[-1],
                    safe_total_rewards[-1],
            )) 
            if n % save_per_episode == 0:
                print("saving results...")
                self.agent.save(fld_model)

        self.visualizer.plot_a_episode(
            self.env, self.agent.model, 
            explored_cum_rewards, explored_actions,
            safe_cum_rewards, safe_actions,
            os.path.join(fld_save, 'episode_%i.png'%(n)))

        self.visualizer.plot_episodes(
            explored_total_rewards, safe_total_rewards, explorations, 
            os.path.join(fld_save, 'total_rewards.png'),
            MA_window)

    def test(self, n_episode, save_per_episode=10, subfld="testing"):

        fld_save = os.path.join(self.fld_save, subfld)
        makedirs(fld_save)
        MA_window = 100  # MA of performance
        safe_total_rewards = []
        path_record = os.path.join(fld_save, "record.csv")

        with open(path_record, "w") as f:
            f.write("episode,game,pnl,rel,MA\n")

        for n in range(n_episode):
            print("\ntesting...")

            safe_cum_rewards, safe_actions, _ = self.play_one_episode(
                0, training=False,
            )
            safe_total_rewards.append(
                100.0 * safe_cum_rewards[-1] / self.env.max_profit
            )
            MA_safe_total_rewards = np.median(safe_total_rewards[-MA_window:])
            ss = [
                str(n),
                self.env.title.replace(",", ";"),
                "%.1f" % (safe_cum_rewards[-1]),
                "%.1f" % (safe_total_rewards[-1]),
                "%.1f" % MA_safe_total_rewards,
            ]

            with open(path_record, "a") as f:
                f.write(",".join(ss) + "\n")
                print("\t".join(ss))

            if n % save_per_episode == 0:
                print("saving results...")

                """
                self.visualizer.plot_a_episode(
                        self.env, self.agent.model, 
                        [np.nan]*len(safe_cum_rewards), [np.nan]*len(safe_actions),
                        safe_cum_rewards, safe_actions,
                        os.path.join(fld_save, 'episode_%i.png'%(n)))

                self.visualizer.plot_episodes(
                        None, safe_total_rewards, None, 
                        os.path.join(fld_save, 'total_rewards.png'),
                        MA_window)
                """

    def __init__(self, agent, env, visualizer, fld_save):

        self.agent = agent
        self.env = env
        self.visualizer = visualizer
        self.fld_save = fld_save


if __name__ == "__main__":
    import tempfile
    from main import get_model
    from sampler import PairSampler
    from emulator import Market
    from agents import Agent

    tmp_dir = tempfile.TemporaryDirectory()

    db = 'randjump_100,1(10, 30)[]_'
    db_type = 'PairSamplerDB'
    fld = os.path.join("..", "data", db_type, db + "A")
    fhr = (10, 30)
    n_section = 1
    max_change_perc = 3.
    noise_level = 5
    game = 'randjump'
    # sampler = PairSampler(
    #     game=game,
    #     window_episode=180,  # 31
    #     forecast_horizon_range=fhr,
    #     n_section=n_section,
    #     noise_level=noise_level,
    #     max_change_perc=max_change_perc,
    #     windows_transform=[],
    # )
    env = Market(
        window_size=31,
        total_days=90,
    )
    model_type = 'conv'
    learning_rate = 1e-4
    discount_factor = 0.99
    batch_size = 8
    fld_load = None
    model, print_t = get_model(model_type, env, learning_rate, fld_load)
    model.model.summary()

    agent = Agent(
        model, discount_factor=discount_factor, batch_size=batch_size
    ) 
    fld_save = tmp_dir.name 
    fld_save = os.path.join('..', 'results', 'tst')
    visualizer = Visualizer(env.action_labels)
    s = Simulator(
        agent=agent,
        env=env,
        visualizer=visualizer,
        fld_save=fld_save,
    )
    # cum_rewards, actions, states = s.play_one_episode(
    #     exploration=0, print_t=True)
    # print('rewards', cum_rewards)
    # print('actions', actions)
    print_t = False
    n_episode_training = 1000
    n_episode_testing = 1
    exploration_decay = 0.999
    exploration_min = 0.05
    exploration_init = 1.0
    print('Start training...')
    s.train(
        n_episode_training,
        save_per_episode=1,
        exploration_decay=exploration_decay,
        exploration_min=exploration_min,
        print_t=print_t,
        exploration_init=exploration_init,
    )
    print('Testing...')
    s.test(
        n_episode_testing, save_per_episode=1, subfld="testing"
    )
    print('Done')
