
# **Playing trading games with deep reinforcement learning**

This repo is the code for this [paper](https://arxiv.org/abs/1803.03916). Deep reinforcement learing is used to find optimal strategies in these two scenarios:
* Momentum trading: capture the underlying dynamics
* Arbitrage trading: utilize the hidden relation among the inputs

Several neural networks are compared: 
* Recurrent Neural Networks (GRU/LSTM)
* Convolutional Neural Network (CNN)
* Multi-Layer Perception (MLP)

### Play with it

`Python 3.6.7` is used to run the code

```bash
$ pyenv activate drl-trader
$ python main.py
```

You can play with model parameters (specified in main.py), if you get good results or any trouble, please contact me at gxiang1228@gmail.com
