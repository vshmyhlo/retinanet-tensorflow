import tensorflow as tf
from network import Network, Sequential

# TODO: fixme
# class Dropout(Network):
#     def __init__(self, dropout_rate, name='dropout'):
#         super().__init__(name=name)
#
#         self.dropout_rate = dropout_rate
#
#     def build(self, input_shape):
#         self.dropout = tf.layers.Dropout(self.dropout_rate)
#
#         super().build(input_shape)
#
#     def call(self, input, training):
#         shape = tf.shape(input)
#         input = self.dropout(input, noise_shape=(shape[0], 1, 1, shape[3]), training=training)
#
#         return input

Dropout = tf.layers.Dropout


class CompositeFunction(Sequential):
    def __init__(self,
                 filters,
                 activation,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='composite_function'):
        layers = [
            tf.layers.BatchNormalization(),
            activation,
            tf.layers.Conv2D(
                filters,
                3,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Dropout(dropout_rate),
        ]

        super().__init__(layers, name=name)


class BottleneckCompositeFunction(Sequential):
    def __init__(self,
                 filters,
                 activation,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='bottleneck_composite_function'):
        layers = [
            tf.layers.BatchNormalization(),
            activation,
            tf.layers.Conv2D(
                filters * 4,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Dropout(dropout_rate),
            tf.layers.BatchNormalization(),
            activation,
            tf.layers.Conv2D(
                filters,
                3,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Dropout(dropout_rate),
        ]

        super().__init__(layers, name=name)


class DenseNet_Block(Network):
    def __init__(self,
                 growth_rate,
                 depth,
                 bottleneck,
                 activation,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='densnet_block'):
        super().__init__(name=name)

        self.composite_functions = []
        for i in range(depth):
            if bottleneck:
                self.composite_functions.append(
                    self.track_layer(
                        BottleneckCompositeFunction(
                            growth_rate,
                            activation=activation,
                            dropout_rate=dropout_rate,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            name='composite_function{}'.format(i + 1))))
            else:
                self.composite_functions.append(
                    self.track_layer(
                        CompositeFunction(
                            growth_rate,
                            activation=activation,
                            dropout_rate=dropout_rate,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            name='composite_function{}'.format(i + 1))))

    def call(self, input, training):
        for f in self.composite_functions:
            output = f(input, training)
            input = tf.concat([input, output], -1)

        return input


class TransitionLayer(Sequential):
    def __init__(self,
                 input_filters,
                 compression_factor,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='transition_layer'):
        self.input_filters = input_filters
        filters = int(input_filters * compression_factor)

        layers = [
            tf.layers.BatchNormalization(),
            tf.layers.Conv2D(
                filters,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Dropout(dropout_rate),
            tf.layers.AveragePooling2D(2, 2, padding='same')
        ]

        super().__init__(layers, name=name)

    def call(self, input, training):
        assert input.shape[-1] == self.input_filters
        return super().call(input, training)


class DenseNetBC_ImageNet(Network):
    def __init__(self,
                 blocks,
                 growth_rate,
                 compression_factor,
                 bottleneck,
                 activation,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='densenet_bc_imagenet'):
        super().__init__(name=name)

        self.conv1 = self.track_layer(
            Sequential([
                tf.layers.Conv2D(
                    2 * growth_rate,
                    7,
                    2,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    name='conv1'),
                tf.layers.BatchNormalization(),
                activation,
            ]))
        self.conv1_max_pool = self.track_layer(
            tf.layers.MaxPooling2D(3, 2, padding='same'))

        self.dense_block_1 = self.track_layer(
            DenseNet_Block(
                growth_rate,
                depth=blocks[1],
                bottleneck=bottleneck,
                activation=activation,
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='dense_block1'))

        self.transition_layer_1 = self.track_layer(
            TransitionLayer(
                input_filters=blocks[1] * growth_rate + 64,
                compression_factor=compression_factor,
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='transition_layer_1'))

        self.dense_block_2 = self.track_layer(
            DenseNet_Block(
                growth_rate,
                depth=blocks[2],
                bottleneck=bottleneck,
                activation=activation,
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='dense_block2'))

        self.transition_layer_2 = self.track_layer(
            TransitionLayer(
                input_filters=blocks[2] * growth_rate + self.transition_layer_1.layers[1].filters,  # FIXME:
                compression_factor=compression_factor,
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='transition_layer_2'))

        self.dense_block_3 = self.track_layer(
            DenseNet_Block(
                growth_rate,
                depth=blocks[3],
                bottleneck=bottleneck,
                activation=activation,
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='dense_block3'))

        self.transition_layer_3 = self.track_layer(
            TransitionLayer(
                input_filters=blocks[3] * growth_rate + self.transition_layer_2.layers[1].filters,  # FIXME:
                compression_factor=compression_factor,
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='transition_layer_3'))

        self.dense_block_4 = self.track_layer(
            DenseNet_Block(
                growth_rate,
                depth=blocks[4],
                bottleneck=bottleneck,
                activation=activation,
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='dense_block4'))

    def call(self, input, training):
        input = self.conv1(input, training)
        C1 = input
        input = self.conv1_max_pool(input)
        input = self.dense_block_1(input, training)
        C2 = input
        input = self.transition_layer_1(input, training)
        input = self.dense_block_2(input, training)
        C3 = input
        input = self.transition_layer_2(input, training)
        input = self.dense_block_3(input, training)
        C4 = input
        input = self.transition_layer_3(input, training)
        input = self.dense_block_4(input, training)
        C5 = input

        return {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}


class DenseNetBC_121(DenseNetBC_ImageNet):
    def __init__(self,
                 activation,
                 dropout_rate,
                 growth_rate=32,
                 compression_factor=0.5,
                 bottleneck=True,
                 name='densenet_bc_121'):
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        super().__init__(
            blocks=[None, 6, 12, 24, 16],
            growth_rate=growth_rate,
            compression_factor=compression_factor,
            bottleneck=bottleneck,
            activation=activation,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name)


class DenseNetBC_169(DenseNetBC_ImageNet):
    def __init__(self,
                 activation,
                 dropout_rate,
                 growth_rate=32,
                 compression_factor=0.5,
                 bottleneck=True,
                 name='densenet_bc_169'):
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        super().__init__(
            blocks=[None, 6, 12, 32, 32],
            growth_rate=growth_rate,
            compression_factor=compression_factor,
            bottleneck=bottleneck,
            activation=activation,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name)
