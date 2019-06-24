import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

from app.lib import DATA_DIR


db_path = os.path.join(DATA_DIR, 'PBSamplerDB', 'uah_to_usd_2018.csv')
df = pd.read_csv(db_path)

df1 = df[:20]
rate = df[['nb']].to_numpy(dtype=np.float32)
scaled_rate = minmax_scale(rate, feature_range=(1, 10))
# import ipdb; ipdb.set_trace()
# print(scaled_rate)
rounded = scaled_rate.round()
# print(rounded)

data = np.reshape(rounded[:, 0], rounded.shape[0])[:30]
x_ticks = np.arange(len(data)+1)

fig = plt.figure()
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

plt.show()
