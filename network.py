import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.estimator import util as estimator_util

tfe_avaiable = hasattr(tfe, 'Network')

if tfe_avaiable:
    Network = tfe.Network
    Sequential = tfe.Sequential
else:
    class Network(tf.layers.Layer):
        def track_layer(self, layer):
            return layer


    class Sequential(Network):
        def __init__(self, layers, name='sequential'):
            super().__init__(name=name)
            self.layers = layers

        def call(self, input, training):
            for layer in self.layers:
                if isinstance(layer, tf.layers.Layer):
                    args = estimator_util.fn_args(layer.call)
                elif isinstance(layer, Network):
                    args = estimator_util.fn_args(layer.call)
                elif callable(layer):
                    args = estimator_util.fn_args(layer)
                else:
                    args = None

                if 'training' in args:
                    input = layer(input, training)
                else:
                    input = layer(input)

            return input
