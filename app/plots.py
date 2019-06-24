import matplotlib.pyplot as plt
import numpy as np


data = [10, 8, 6, 4, 2, 1, 9, 6, 6, 4]
x_ticks = np.arange(len(data)+1)

y_ticks = np.arange(1, 11, step=1)
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
plt.bar(np.arange(len(data)), data, align='edge', width=1., color='tab:pink')
plt.ylabel('Normalized price')
plt.yticks(y_ticks)
plt.xticks(x_ticks)
plt.title('Converted rates data')
plt.grid(True)

plt.show()
