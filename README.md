
# **Playing trading games with deep reinforcement learning**

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
```

### Play with it
Just call the main function

    python main.py

* Testing data generation with sampler

```bash
$ python app/sampler.py
```

You can play with model parameters (specified in main.py), if you get good results or any trouble, please contact me at gxiang1228@gmail.com
