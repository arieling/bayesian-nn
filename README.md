# bayesian-nn
bayesian-nn is a lightweight *Bayesian neural network* library built on top of tensorflow where training is completed with *stochastic variational inference* (SVI). The library is intended to resemble [tf.slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) and help avoid massive boilerplate code. The end goal is to facilitate speedy development of Bayesian neural net models in the case where multiple stacked layers are required.

## Installation
```bash
pip install bayesian-nn
```

## Usage
```python
import bayesian-nn as bnn
```

## How are Bayesian neural nets trained with SVI?
![](assets/bbb_demo.gif)

## Layers
bayesian-nn primarily provides the user with the flexibility of stacking neural net layers where weight distributions are trained through SVI.

Pre-implemented layers include:

Layer | bayesian-nn
------- | --------
FullyConnected | [bnn.fully_connected]()
Conv2d | [bnn.conv2d]()
Conv2dTranspose (Deconv) | [bnn.conv2d_transpose]()
RNN | [bnn.rnn]()

## Features
The user can also further simplify boilerplate code through the following side features:

* [arg_scope](): allow users to define default arguments for specific operations within that scope.
* [repeat](): allow users to repeatedly perform the same operation with the same parameters.
* [stack](): allow users to perform the same operation with different parameters.

## References
