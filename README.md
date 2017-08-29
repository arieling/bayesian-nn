# BayesianNN
BayesianNN is a lightweight Bayesian neural network library built on top of tensorflow where network training is completed through stochastic variational inference (SVI). The library mainly facilitates speedy development of Bayesian neural net models in the case where multiple stacked layers are required.

## Installation
```bash
pip install bayesian-nn
```

## Usage
```python
import bayesian-nn as bnn
```

## How are Bayesian neural nets trained with SVI?


## Layers
BayesianNN primarily provides the user with the flexibility of stacking neural net layers where weight distributions are trained through SVI.

Pre-implemented layers include:

Layer | BayesianNN
------- | --------
FullyConnected | [bnn.fully_connected]()
Conv2dInPlane | [bnn.conv2d]()
Conv2dTranspose (Deconv) | [bnn.conv2d_transpose]()
RNN | [bnn.rnn]()

## Other Features
Although BayesianNN mainly provides pre-implemented layers where weights are trained to follow certain distributions, the user can also tweak the detailed settings. 

## References
