import tensorflow as tf


class GroupNormalization(tf.layers.Layer):
    def __init__(self, G=32, eps=1e-5, name='group_normalization'):
        super().__init__(name=name)

        self._G = G
        self._eps = eps

    def build(self, input_shape):
        C = input_shape[-1]

        # per channel gamma and beta
        self._gamma = self.add_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        self._beta = self.add_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

        super().build(input_shape)

    def call(self, input):
        N, H, W, _ = tf.unstack(tf.shape(input))
        _, _, _, C = input.shape

        G = min(self._G, C)

        # add groups
        input = tf.reshape(input, [N, H, W, G, C // G])

        # normalize
        mean, var = tf.nn.moments(input, [1, 2, 4], keep_dims=True)
        input = (input - mean) / tf.sqrt(var + self._eps)

        input = tf.reshape(input, [N, H, W, C]) * self._gamma + self._beta

        return input


Normalization = GroupNormalization
