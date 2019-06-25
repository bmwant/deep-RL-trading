from typing import List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    """
    https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    """
    def samplemat(dims):
        import random
        aa = np.zeros(dims)
        for i in range(30):
            k = random.randint(0, 2)
            aa[k, i] = k+1

        return aa

    cmap = mcolors.ListedColormap(['w', 'tab:green', 'tab:red', 'tab:cyan'])
    # Display matrix
    ax = plt.matshow(
        samplemat((3, 30)),
        cmap=cmap,
        extent=(0, 30, 3, 0),  # (left, right, bottom, top)
    )

    actions = ['sell', 'buy', 'hold']
    plt.xticks(np.arange(30))
    plt.yticks(np.arange(len(actions)), actions)
    import ipdb; ipdb.set_trace()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_actions(data: List[int]):
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
