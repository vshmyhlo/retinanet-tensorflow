import tensorflow as tf
from network import Network


class GroupNormalization(Network):
    def __init__(self, G=32, epsilon=1e-5, name='group_normalization'):
        super().__init__(name=name)

        self.G = G
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = tf.get_variable('gamma', [input_shape[-1]], initializer=tf.constant_initializer(1.0))
        self.beta = tf.get_variable('beta', [input_shape[-1]], initializer=tf.constant_initializer(0.0))

        super().build(input_shape)

    def call(self, input, training):
        # transpose: [bs, h, w, c] to [bs, c, h, w] following the paper
        input = tf.transpose(input, [0, 3, 1, 2])

        input_shape = tf.shape(input)
        N, C, H, W = (input_shape[i] for i in range(4))
        G = tf.minimum(self.G, C)

        input = tf.reshape(input, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(input, [2, 3, 4], keep_dims=True)
        input = (input - mean) / tf.sqrt(var + self.epsilon)

        # per channel gamma and beta
        gamma = tf.reshape(self.gamma, [1, C, 1, 1])
        beta = tf.reshape(self.beta, [1, C, 1, 1])

        input = tf.reshape(input, [N, C, H, W]) * gamma + beta
        # transpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        input = tf.transpose(input, [0, 2, 3, 1])

        return input
