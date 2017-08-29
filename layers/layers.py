from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import Normal
from tensorflow.contrib.layers import xavier_initializer

xi = xavier_initializer
ni = tf.random_normal_initializer
eps = 1e-35  # small epsilon avoid -inf


def gaussian_layer(x,
                   in_dim,
                   out_dim,
                   scope,
                   activation_fn=tf.nn.relu,
                   reuse=False,
                   use_mean=False,
                   store=False,
                   use_stored=False,
                   prior_stddev=1.0,
                   l2_const=0.0):
    """Single layer of fully-connected units where the weights follow a
    unit gaussian prior, and
    Args:
        x: batch of input
        in_dim: input dimension
        out_dim: output dimension
        scope: tensorflow variable scope name
        activation_fn: activation function
        use_mean: use the mean of approximate posterior, instead of sampling
        closed_form_kl: return closed form kl
    Returns:
        output and kl of the weights for the layer
    """

    prior_var = prior_stddev ** 2

    with tf.variable_scope(scope, reuse=reuse):

        w_mean = tf.get_variable('w_mean', shape=[in_dim, out_dim], initializer=xi())
        w_row = tf.get_variable('w_row', shape=[in_dim, out_dim], initializer=ni(-3.0, 0.1))
        w_stddev = tf.nn.softplus(w_row, name='w_std') + eps
        w_dist = Normal([0.0]*in_dim*out_dim, [1.0]*in_dim*out_dim)
        w_std_sample = tf.reshape(w_dist.sample(), [in_dim, out_dim], name='w_std_sample')

        # local reparametrization
        w_sample = w_mean + w_std_sample * w_stddev
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=xi())

        # to store the previous theta value
        w_last = tf.get_variable('w_last', initializer=tf.zeros([in_dim, out_dim]), trainable=False)

        if use_mean:
            out = activation_fn(tf.matmul(x, w_mean) + b, name='activation')
            return out, 0.0
        else:
            if store:
                store_op = tf.assign(w_last, w_sample)
                with tf.control_dependencies([store_op]):
                    out = activation_fn(tf.matmul(x, w_sample) + b, name='activation')
            else:
                if use_stored:
                    out = activation_fn(tf.matmul(x, w_last) + b, name='activation')
                else:
                    out = activation_fn(tf.matmul(x, w_sample) + b, name='activation')

            D = in_dim * out_dim
            kl = tf.log(prior_stddev) * D - \
                 tf.reduce_sum(tf.log(w_stddev+eps)) + \
                 0.5*(-D +
                     (tf.reduce_sum(w_stddev**2) +
                      tf.reduce_sum(w_mean**2)) / prior_var)
            return out, kl


def main():
    pass

if __name__ == '__main__':
    main()
