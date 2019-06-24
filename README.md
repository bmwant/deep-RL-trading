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

### Authors

* gxiang1228@gmail.com
