import tensorflow as tf
from tensorflow.python.util import tf_inspect


class Model(tf.layers.Layer):
    def __init__(self, name='model'):
        super().__init__(name=name)


class Sequential(Model):
    def __init__(self, layers, name='sequential'):
        super().__init__(name=name)

        self.layers = layers

    def call(self, input, training):
        for layer in self.layers:
            if isinstance(layer, tf.layers.Layer) and 'training' in tf_inspect.getargspec(layer.call).args:
                input = layer(input, training=training)
            elif callable(layer) and 'training' in tf_inspect.getargspec(layer).args:
                input = layer(input, training=training)
            else:
                input = layer(input)

        return input
