import tensorflow as tf


class GroupNormalization(tf.layers.Layer):
    def __init__(self, groups=32, eps=1e-5, name='group_normalization'):
        super().__init__(name=name)

        self.groups = groups
        self.eps = eps

    def build(self, input_shape):
        c = input_shape[-1]

        # per channel gamma and beta
        self.gamma = self.add_variable('gamma', [1, 1, 1, c], initializer=tf.constant_initializer(1.0))
        self.beta = self.add_variable('beta', [1, 1, 1, c], initializer=tf.constant_initializer(0.0))

        super().build(input_shape)

    def call(self, input):
        n, h, w, _ = tf.unstack(tf.shape(input))
        _, _, _, c = input.shape

        groups = min(self.groups, c)

        # add groups
        input = tf.reshape(input, [n, h, w, groups, c // groups])

        # normalize
        mean, var = tf.nn.moments(input, [1, 2, 4], keep_dims=True)
        input = (input - mean) / tf.sqrt(var + self.eps)

        input = tf.reshape(input, [n, h, w, c]) * self.gamma + self.beta

        return input


# TODO: use best implementation by default
class Normalization(GroupNormalization):
    def call(self, input, training):
        return super().call(input)
