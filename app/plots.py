from typing import List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def plot_state_prices_window(data):
    x_ticks = np.arange(0, len(data) + 1, step=1)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    y_ticks = np.arange(1, 11, step=1)
    # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    plt.bar(np.arange(len(data)), data, align='edge', width=1., color='tab:pink')
    plt.ylabel('Normalized price')
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.title('Converted rates data')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_episode_chart(
    *,
    episode: int = 0,
    safe_actions: List[int],
    safe_rewards: List[float],
    explored_rewards: List[float],
    exploration: float,
    extra: dict,
    ma_window: int = 30,
    save_path=None,
):
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)
    fig = plt.figure(figsize=(10, 8))
    x_ticks = np.arange(300, step=15)
    ax1 = fig.add_subplot(grid[0, :])
    ax1.set_title('Episode #%d. MA window: %d' % (episode, ma_window))
    ax1.axhline(y=0, color='lightgrey', linestyle='--')
    ax1.set_xticks(x_ticks)
    # Safe rewards and moving average
    ax1.plot(safe_rewards,
             label='safe rewards', color='mediumaquamarine', linestyle=':')
    ma_safe = pd.Series(np.array(safe_rewards)) \
        .rolling(window=ma_window, min_periods=1).median()
    ax1.plot(ma_safe, label='MA safe', color='mediumseagreen')
    # Explored rewards and moving average
    ax1.plot(explored_rewards,
             label='explored rewards', linestyle=':', color='lightsalmon')
    ma_explored = pd.Series(np.array(explored_rewards)) \
        .rolling(window=ma_window, min_periods=1).median()
    ax1.plot(ma_explored, label='MA explored', color='orangered')
    ax1.legend()

    ax2 = fig.add_subplot(grid[1, 0])
    ax2.axis([0, 10, 0, 10])
    ax2.axis('off')
    ax2.set_title('Episode data')
    ax2.text(0, 9, 'Total profit (safe): %.2f' % extra['profit_safe'])
    ax2.text(0, 8, 'Total profit (explored): %.2f' % extra['profit_explored'])
    ax2.text(0, 7, 'Total reward (safe): %.2f' % extra['reward_safe'])
    ax2.text(0, 6, 'Total reward (explored): %.2f' % extra['reward_explored'])
    ax2.text(0, 5, 'Exploration: %.2f' % exploration)

    ax3 = fig.add_subplot(grid[1, 1])
    ax3.set_title('Actions distribution')
    actions = np.array(safe_actions)
    actions = actions[~np.isnan(actions)].astype(np.int32)
    actions_bins = np.bincount(actions)
    actions_labels = ['sell', 'buy', 'hold']
    ax3.set_xticks(np.arange(len(actions_labels)))
    ax3.set_xticklabels(actions_labels)
    ax3.bar(
        np.arange(len(actions_bins)), actions_bins, align='center', width=0.8)

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def show_episodes_chart(
    *,
    explorations: List[float],
):
    """
    https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html#plt.GridSpec:-More-Complicated-Arrangements
    """
    grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.3)
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(grid[0, :])

    ax2 = fig.add_subplot(grid[1, 0])
    ax2.set_title('Exploration')
    ax2.plot(explorations)

    ax3 = fig.add_subplot(grid[1, 1])
    ax3.set_title('Actions distribution')
    ax3.bar(
        np.arange(3), [7, 8, 2], align='center', width=0.8)
    actions = ['sell', 'buy', 'hold']
    ax3.set_xticks(np.arange(len(actions)))
    ax3.set_xticklabels(actions)

    ax4 = fig.add_subplot(grid[2, 0])
    ax5 = fig.add_subplot(grid[2, 1])

    data = [5, 3, 2, 8, 9, 10]
    ax1.plot(data)

    ax5.plot(data)
    plt.tight_layout()
    plt.show()


def show_step_chart(
    *,
    prices,
    slots,
    actions,
    step: int = 0,
    window_state: int = 30,
    save_path=None,
):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1,
        sharex=False, sharey=False,
        figsize=(10, 8),
    )
    fig.canvas.set_window_title('Episode #0. Position #%d' % step)

    _show_prices(ax1, prices=prices, step=step, window_state=window_state)

    # what does agent see
    state = prices[step-window_state:step].transpose()[0]
    _show_state(ax2, state=state, step=step)

    _show_slots(ax3, slots=slots)

    _show_actions(ax4, actions=actions, step=step)

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def _show_prices(ax, prices, step: int = 0, window_state: int = 0):
    ax.set_title('Prices (%d days)' % len(prices))
    ax.plot(prices)
    ax_xticks = np.arange(0, len(prices) + 1, step=30)
    ax.set_xticks(ax_xticks)
    ax_yticks = np.arange(0, 11, step=2)  # max price parametrized
    ax.set_yticks(ax_yticks)
    # Build vertical lines to show current state window
    ax.axvline(x=step - window_state, color='m', linestyle=':')
    ax.axvline(x=step, color='m', linestyle=':')


def _show_state(ax, state, step: int = 0):
    window_state = len(state)
    ax.set_title('State (window: %d days)' % window_state)
    i_start = step - window_state
    i_end = step + 1
    ax.set_xticklabels(np.arange(i_start, i_end, step=5))
    ax_xticks = np.arange(0, window_state+1, step=5)
    ax_minor_xticks = np.arange(0, window_state+1, step=1)
    ax.set_xticks(ax_xticks)
    ax.set_xticks(ax_minor_xticks, minor=True)
    ax.set_ylim(0, 11)
    ax_yticks = np.arange(0, 11, step=1)  # max price parametrized
    ax.set_yticks(ax_yticks)

    ax.bar(
        np.arange(len(state)), state, align='edge', width=1., color='tab:pink')
    ax.grid(True, which='both')
    ax.set_aspect('equal')


def _show_slots(ax, slots):
    ax.set_title('Slots (max size: 10)')
    slots_labels = ['price']
    im = ax.imshow(slots, cmap='RdPu')
    ax.set_xticks(np.arange(len(slots[0])))
    ax.set_yticks(np.arange(len(slots_labels)))
    ax.set_yticklabels(slots_labels)
    # Create text annotations with prices
    for i in range(len(slots[0])):
        text = '%.2f' % slots[0, i]
        color = 'w'
        if not slots[0, i]:
            text = 'empty'
            color = 'k'
        ax.text(i, 0, text, ha='center', va='center', color=color)


def _show_actions(ax, actions: List[int], step: int = 0):
    cols = 30  # show last actions
    step = max(step, cols)  # make sure we can slice this window
    ax.grid(True)
    ax_ticks = np.arange(0, cols + 1, step=1)
    ax.set_xticks(ax_ticks)
    ax.set_xticklabels(np.arange(step-cols, step, step=1))
    actions_labels = ['sell', 'buy', 'hold']
    ax.set_yticks(np.arange(len(actions_labels)))
    ax.set_yticklabels(actions_labels)

    rows = len(actions_labels)
    ax.set_title('Actions (recent %d)' % cols)
    matrix = np.zeros((rows, cols))
    actions = actions[step-cols:step]  # todo (misha): be careful
    for i, action in enumerate(actions):
        if not np.isnan(action):
            matrix[action, i] = action + 1

    cmap = mcolors.ListedColormap(['w', 'tab:green', 'tab:red', 'tab:cyan'])
    im = ax.imshow(
        matrix,
        cmap=cmap,
        extent=(0, cols, rows, 0),
    )


def show_actions(data: List[int]):
    """
    https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    """
    SHOW_LAST_N = 30
    # todo (misha): this is wrong, we should map values to colors
    cmap = mcolors.ListedColormap(['w', 'tab:green', 'tab:red', 'tab:cyan'])
    actions = ['sell', 'buy', 'hold']
    data = data[-SHOW_LAST_N:]
    rows, cols = (len(actions), len(data))
    matrix = np.zeros((rows, cols))
    for i, action in enumerate(data):
        matrix[action, i] = action+1

    # Display matrix
    plt.matshow(
        matrix,
        cmap=cmap,
        extent=(0, cols, rows, 0),  # (left, right, bottom, top)
    )

    plt.xticks(np.arange(cols))
    plt.yticks(np.arange(len(actions)), actions)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    pass


if __name__ == '__main__':
    main()
