from typing import List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import numpy as np


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


def main():
    x = [1 ,3,5,3, 4,5, 6,23, 2,3]
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(
        4, 1,
        sharex=False, sharey=False,
        figsize=(10, 8),
    )
    fig.canvas.set_window_title('Episode #0')

    ax1.set_title('Prices (300 days)')
    ax1.plot(x)


    ax2.set_title('State (window: 30 days)')
    ax2.grid(True)
    ax2_ticks = np.arange(0, 30 + 1, step=1)
    ax2.set_xticks(ax2_ticks)


    ax3.set_title('Slots (max size: 10)')

    ax4.set_title('Actions (last 30)')
    ax4.grid(True)
    ax4_ticks = np.arange(0, 30 + 1, step=1)
    ax4.set_xticks(ax4_ticks)
    actions = ['sell', 'buy', 'hold']
    ax4.set_yticks(np.arange(len(actions)))
    ax4.set_yticklabels(actions)

    plt.tight_layout()
    plt.show()


def show_episode(
        prices,
        step: int = 0,
        window_state: int = 30,
):
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(
        4, 1,
        sharex=False, sharey=False,
        figsize=(10, 8),
    )
    fig.canvas.set_window_title('Episode #0')

    ax1.set_title('Prices (%d days)' % len(prices))
    ax1.plot(prices)
    ax1_xticks = np.arange(0, len(prices)+1, step=30)
    ax1.set_xticks(ax1_xticks)
    ax1_yticks = np.arange(0, 11, step=2)  # max price parametrized
    ax1.set_yticks(ax1_yticks)
    # Build vertical lines to show current state window
    ax1.axvline(x=step, color='m', linestyle=':')
    ax1.axvline(x=step+window_state, color='m', linestyle=':')

    ax2.set_title('State (window: %d days)' % window_state)

    ax2_xticks = np.arange(step, step+window_state+1, step=5)
    ax2_minor_xticks = np.arange(step, step+window_state+1, step=1)
    ax2.set_xticks(ax2_xticks)
    ax2.set_xticks(ax2_minor_xticks, minor=True)
    ax2_yticks = np.arange(0, 11, step=1)  # max price parametrized
    ax2.set_yticks(ax2_yticks)
    state = prices[step:step+window_state].transpose()[0]
    ax2.bar(
        np.arange(len(state)), state, align='edge', width=1., color='tab:pink')
    ax2.grid(True, which='both')
    ax2.set_aspect('equal')

    ax3.set_title('Slots (max size: 10)')

    ax4.set_title('Actions (last 30)')
    ax4.grid(True)
    ax4_ticks = np.arange(0, 30 + 1, step=1)
    ax4.set_xticks(ax4_ticks)
    actions = ['sell', 'buy', 'hold']
    ax4.set_yticks(np.arange(len(actions)))
    ax4.set_yticklabels(actions)

    plt.tight_layout()
    plt.show()


def show_actions(data: List[int]):
    """
    https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    """
    SHOW_LAST_N = 30
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


if __name__ == '__main__':
    main()
