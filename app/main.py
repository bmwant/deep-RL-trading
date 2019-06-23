import os

import click

from app.lib import OUTPUT_FLD, ROOT_DIR
from app.sampler import PairSampler, SinSampler
from app.visualizer import Visualizer
from app.emulator import Market
from app.simulators import Simulator
from app.agents import load_model
from app.agents import (
    Agent,
    QModelConv,
    QModelMLP,
    QModelGRU,
    QModelConvGRU,
)


def get_model(model_type, env, learning_rate, fld_load=None):
    if model_type == 'MLP':
        m = 16
        layers = 5
        hidden_size = [m]*layers
        model = QModelMLP(env.state_shape, env.n_action)
        model.build_model(
            hidden_size,
            learning_rate=learning_rate,
            activation='tanh',
        )
    elif model_type == 'conv':
        m = 16
        layers = 2
        filter_num = [m]*layers
        filter_size = [3] * len(filter_num)
        #use_pool = [False, True, False, True]
        #use_pool = [False, False, True, False, False, True]
        use_pool = None
        #dilation = [1,2,4,8]
        dilation = None
        dense_units = [48, 24]
        model = QModelConv(env.state_shape, env.n_action)
        model.build_model(
            filter_num,
            filter_size,
            dense_units,
            learning_rate,
            dilation=dilation,
            use_pool=use_pool,
        )
    elif model_type == 'RNN':
        m = 32
        layers = 3
        hidden_size = [m]*layers
        dense_units = [m, m]
        model = QModelGRU(env.state_shape, env.n_action)
        model.build_model(hidden_size, dense_units, learning_rate=learning_rate)
    elif model_type == 'ConvRNN':
        m = 8
        conv_n_hidden = [m, m]
        RNN_n_hidden = [m, m]
        dense_units = [m, m]
        model = QModelConvGRU(env.state_shape, env.n_action)
        model.build_model(
            conv_n_hidden,
            RNN_n_hidden,
            dense_units,
            learning_rate=learning_rate,
        )
    elif model_type == 'pretrained':
        model = load_model(fld_load, learning_rate)
    else:
        raise ValueError('Incorrect model type was selected')

    return model


def main():
    model_type = 'conv'  # default model type
    fld_load = None
    n_episode_training = 10  # number of training episodes
    n_episode_testing = 20  # number of testing episodes
    open_cost = 3.3  # cost of opening a bid
    univariate = True
    # which data to use when training
    if univariate:
        db_type = 'SinSamplerDB'
        db = 'concat_half_base_'
        Sampler = SinSampler
    else:
        db_type = 'PairSamplerDB'
        db = 'randjump_100,1(10, 30)[]_'
        Sampler = PairSampler
    # directory for the data to load from
    fld = os.path.join(ROOT_DIR, 'data', db_type, db+'A')
    # load data from directory specified
    sampler = Sampler('load', fld=fld)

    # RL-related settings
    batch_size = 8
    learning_rate = 1e-4
    discount_factor = 0.8
    exploration_init = 1.
    exploration_decay = 0.99
    exploration_min = 0.01
    window_state = 40

    # create environment (market) from the data loaded
    env = Market(sampler, window_state, open_cost)
    # create a model based on type selected
    model = get_model(model_type, env, learning_rate, fld_load)
    model.model.summary()

    # create an RL agent
    agent = Agent(model, discount_factor=discount_factor, batch_size=batch_size)
    visualizer = Visualizer(env.action_labels)

    # output directory to save intermediate results
    fld_save = os.path.join(
        OUTPUT_FLD, sampler.title, model.model_name,
        str((env.window_state, sampler.window_episode, agent.batch_size,
             learning_rate, agent.discount_factor, exploration_decay,
             env.open_cost)))

    print('='*20)
    print(fld_save)
    print('='*20)

    # train a model
    simulator = Simulator(agent, env, visualizer=visualizer, fld_save=fld_save)
    simulator.train(
        n_episode_training,
        save_per_episode=1,  # save data on each episode
        exploration_init=exploration_init,
        exploration_decay=exploration_decay,
        exploration_min=exploration_min,
    )

    #agent.model = load_model(os.path.join(fld_save,'model'), learning_rate)

    if univariate:
        print('Testing trained model for univariate (sin-like) data')
        fld = os.path.join(ROOT_DIR, 'data', db_type, db+'B')
        sampler = SinSampler('load', fld=fld)
        simulator.env.sampler = sampler
        simulator.test(
            n_episode_testing,
            save_per_episode=1,
            subfld='out-of-sample testing',
        )
    else:
        print('Testing trained model for bivariate (rand-jump) data')
        simulator.test(
            n_episode_testing,
            save_per_episode=1,
            subfld='in-sample testing',
        )


def custom_launch():
    from app.sampler import PBSampler

    model_type = 'conv'
    n_episode_training = 300
    n_episode_testing = 1
    open_cost = 0.1

    sampler = PBSampler()
    window_state = 30  # set to month by default
    learning_rate = 1e-4
    discount_factor = 0.95
    batch_size = 8

    exploration_init = 1.  # always explore at the beginning
    exploration_decay = 0.99
    exploration_min = 0.01
    ma_window = 60  # just to measure overall performance

    env = Market(
        sampler=sampler,
        window_state=window_state,
        open_cost=open_cost,
    )
    model = get_model(
        model_type=model_type,
        env=env,
        learning_rate=learning_rate,
    )

    fld_save = os.path.join(
        OUTPUT_FLD, 'PB_2018_180d_30s_test5'
    )

    # fld_load_model = os.path.join(fld_save, 'model')
    # model = get_model(
    #     model_type='pretrained',
    #     env=env,
    #     learning_rate=learning_rate,
    #     fld_load=fld_load_model,
    # )
    model.model.summary()

    agent = Agent(
        model=model,
        discount_factor=discount_factor,
        batch_size=batch_size,
    )

    visualizer = Visualizer(env.action_labels)

    # env.sampler.offset = 90
    simulator = Simulator(
        agent=agent,
        env=env,
        visualizer=visualizer,
        fld_save=fld_save,
        ma_window=ma_window,
    )

    click.secho('Training agent...', fg='green')
    simulator.train(
        n_episode=n_episode_training,
        save_per_episode=1,
        exploration_init=exploration_init,
        exploration_decay=exploration_decay,
        exploration_min=exploration_min,
    )

    click.secho('Testing agent...', fg='green')

    simulator.test(
        n_episode=n_episode_testing,
        save_per_episode=1,
        subfld='in-sample-testing',
        verbose=True,
    )


if __name__ == '__main__':
    # main()
    custom_launch()
