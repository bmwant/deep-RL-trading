## Playing trading games with deep RL

This repo is the code for this [paper](https://arxiv.org/abs/1803.03916).
Deep reinforcement learning is used to find optimal strategies in these two scenarios:
* Momentum trading: capture the underlying dynamics
* Arbitrage trading: utilize the hidden relation among the inputs

Several neural networks are compared: 
* Recurrent Neural Networks (GRU/LSTM)
* Convolutional Neural Network (CNN)
* Multi-Layer Perception (MLP)

### Dependencies

```bash
$ pyenv local 3.6.7
$ pyenv activate drl-trader
$ pip install -r requirements.txt
$ pip install -e .
```

Set matplotlib backend `~/.matplotlib/matplotlibrc`

```text
backend: qt5agg
```

### Play with it

* Launch main script

```bash
$ python app/main.py
```

* Testing data generation with sampler

```bash
$ python app/sampler.py
```

* Leave agent training for a long period of time

```bash
$ caffeinate -sid python app/main.py
```


### Utils

* Convert data (scale within a range)

```bash
$ python app/converter.py data/PBSamplerDB/uah_to_usd_2018.csv \
    data/uah_to_usd_2018_scaled_1_10.csv
```

### Authors

* gxiang1228@gmail.com
