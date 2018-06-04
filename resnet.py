import tensorflow as tf
from network import Network, Sequential


# TODO: check initialization
# TODO: check regularization
# TODO: check resize-conv (upsampling)
# TODO: check training arg
# TODO: remove bias where not needed


class ResNeXt_Bottleneck(Network):
    def __init__(self, filters_base, project, kernel_initializer, kernel_regularizer, cardinality=32,
                 name='resnext_bottleneck'):
        assert filters_base % cardinality == 0
        assert project in [True, False, 'down']
        super().__init__(name=name)

        # identity
        if project == 'down':
            self.identity = self.track_layer(
                Sequential([
                    # TODO: check this
                    tf.layers.Conv2D(
                        filters_base * 4, 2, 2, padding='same', use_bias=False, kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer),
                    tf.layers.BatchNormalization()
                ]))
        elif project:
            self.identity = self.track_layer(
                Sequential([
                    tf.layers.Conv2D(
                        filters_base * 4, 1, use_bias=False, kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer),
                    tf.layers.BatchNormalization()
                ]))
        else:
            self.identity = None

        # conv1
        self.conv1 = self.track_layer(Sequential([
            tf.layers.Conv2D(
                filters_base * 2, 1, use_bias=False, kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tf.layers.BatchNormalization(),
            tf.nn.relu
        ]))

        # conv2
        self.conv2 = []
        for i in range(cardinality):
            strides = 2 if project == 'down' else 1

            conv = self.track_layer(
                Sequential([
                    tf.layers.Conv2D(
                        (filters_base * 2) // cardinality, 3, strides, padding='same', use_bias=False,
                        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer),
                    tf.layers.BatchNormalization(),
                    tf.nn.relu
                ]))

            self.conv2.append(conv)

        # conv3
        self.conv3 = self.track_layer(Sequential([
            tf.layers.Conv2D(
                filters_base * 4, 1, use_bias=False, kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tf.layers.BatchNormalization()
        ]))

    def call(self, input, training):
        if self.identity is not None:
            identity = self.identity(input, training)
        else:
            identity = input

        # conv1
        input = self.conv1(input, training)

        # conv2
        splits = tf.split(input, len(self.conv2), -1)
        assert len(splits) == len(self.conv2)
        transformations = []
        for trans, conv in zip(splits, self.conv2):
            trans = conv(trans, training)
            transformations.append(trans)
        input = tf.concat(transformations, -1)

        # conv3
        input = self.conv3(input, training)
        input = input + identity
        input = tf.nn.relu(input)

        return input


class ResNeXt_Block(Sequential):
    def __init__(self, filters_base, depth, downsample, kernel_initializer, kernel_regularizer, name='resnext_block'):
        layers = []

        for i in range(depth):
            if i == 0:
                project = 'down' if downsample else True
            else:
                project = False

            layer = ResNeXt_Bottleneck(
                filters_base, project=project, kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer, name='conv{}'.format(i + 1))
            layers.append(layer)

        super().__init__(layers, name=name)


class ResNeXt_Conv1(Network):
    def __init__(self, kernel_initializer, kernel_regularizer, name='resnext_conv1'):
        super().__init__(name=name)

        self.conv = self.track_layer(
            tf.layers.Conv2D(
                64, 7, 2, padding='same', use_bias=False, kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer, name='conv1'))
        self.bn = self.track_layer(tf.layers.BatchNormalization())

    def call(self, input, training):
        input = self.conv(input)
        input = self.bn(input, training)
        input = tf.nn.relu(input)

        return input


class ResNeXt(Network):
    def __init__(self, kernel_initializer, kernel_regularizer, name='resnext'):
        super().__init__(name=name)

        self.conv1 = self.track_layer(
            ResNeXt_Conv1(kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name='conv1'))
        self.conv1_max_pool = self.track_layer(tf.layers.MaxPooling2D(3, 2, padding='same'))

    def call(self, input, training):
        input = self.conv1(input, training)
        C1 = input
        input = self.conv1_max_pool(input)
        input = self.conv2(input, training)
        C2 = input
        input = self.conv3(input, training)
        C3 = input
        input = self.conv4(input, training)
        C4 = input
        input = self.conv5(input, training)
        C5 = input

        return {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}


class ResNeXt_50(ResNeXt):
    def __init__(self, name='resnext_v2_50'):
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        super().__init__(kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name)

        self.conv2 = self.track_layer(
            ResNeXt_Block(
                filters_base=64, depth=3, downsample=False, kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer, name='conv2'))
        self.conv3 = self.track_layer(
            ResNeXt_Block(
                filters_base=128, depth=4, downsample=True, kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer, name='conv3'))
        self.conv4 = self.track_layer(
            ResNeXt_Block(
                filters_base=256, depth=6, downsample=True, kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer, name='conv4'))
        self.conv5 = self.track_layer(
            ResNeXt_Block(
                filters_base=512, depth=3, downsample=True, kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer, name='conv5'))
