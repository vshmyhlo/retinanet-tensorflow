import tensorflow as tf
from network import Network, Sequential

# TODO: check input normalization
# TODO: check outputs followed by activation


class CompositeFunction(Sequential):
    def __init__(self,
                 filters,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='composite_function'):
        layers = [
            tf.layers.BatchNormalization(),
            tf.nn.relu,
            tf.layers.Conv2D(
                filters,
                3,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tf.layers.Dropout(dropout_rate),
        ]

        super().__init__(layers, name=name)


class BottleneckCompositeFunction(Sequential):
    def __init__(self,
                 filters,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='bottleneck_composite_function'):
        layers = [
            tf.layers.BatchNormalization(),
            tf.nn.relu,
            tf.layers.Conv2D(
                filters * 4,
                1,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tf.layers.Dropout(dropout_rate),
            tf.layers.BatchNormalization(),
            tf.nn.relu,
            tf.layers.Conv2D(
                filters,
                3,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tf.layers.Dropout(dropout_rate),
        ]

        super().__init__(layers, name=name)


class DenseNet_Block(Network):
    def __init__(self,
                 growth_rate,
                 depth,
                 bottleneck,
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
                            dropout_rate=dropout_rate,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            name='composite_function{}'.format(i + 1))))
            else:
                self.composite_functions.append(
                    self.track_layer(
                        CompositeFunction(
                            growth_rate,
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
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tf.layers.Dropout(dropout_rate),
            tf.layers.AveragePooling2D(2, 2)
        ]

        super().__init__(layers, name=name)

    def call(self, input, training):
        assert input.shape[-1] == self.input_filters
        return super().call(input, training)


class DenseNetBC_ImageNet(Network):
    def __init__(self, growth_rate, name='densenet_bc_imagenet'):
        super().__init__(name=name)

        self.conv1 = self.track_layer(
            Sequential([
                tf.layers.Conv2D(
                    2 * growth_rate, 7, 2, padding='same', name='conv1'),
                tf.layers.BatchNormalization(),
                tf.nn.relu,
            ]))
        self.conv1_max_pool = self.track_layer(
            tf.layers.MaxPooling2D(3, 2, padding='same'))

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


class DenseNetBC_169(DenseNetBC_ImageNet):
    def __init__(self,
                 dropout_rate,
                 growth_rate=32,
                 compression_factor=0.5,
                 bottleneck=True,
                 name='densenet_bc_169'):  # TODO: bottleneck to true
        super().__init__(growth_rate, name=name)

        blocks = [None, 6, 12, 32, 32]
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        self.dense_block_1 = self.track_layer(
            DenseNet_Block(
                growth_rate,
                depth=blocks[1],
                bottleneck=bottleneck,
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
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='dense_block2'))

        self.transition_layer_2 = self.track_layer(
            TransitionLayer(
                input_filters=blocks[2] * growth_rate +
                self.transition_layer_1.layers[1].filters,
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
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='dense_block3'))

        self.transition_layer_3 = self.track_layer(
            TransitionLayer(
                input_filters=blocks[3] * growth_rate +
                self.transition_layer_2.layers[1].filters,
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
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='dense_block4'))
