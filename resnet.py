import tensorflow as tf
from normalization import Normalization


# TODO: make baseclass
# TODO: use enum for downsample type
# TODO: remove redundant `name`
# TODO: check initialization
# TODO: check regularization
# TODO: check resize-conv (upsampling)
# TODO: check training arg
# TODO: remove bias where not needed


class ResNeXt_Bottleneck(tf.layers.Layer):
    def __init__(
            self, filters, project, kernel_initializer, kernel_regularizer, cardinality=32, name='resnext_bottleneck'):
        assert filters % cardinality == 0
        assert project in [True, False, 'down']

        super().__init__(name=name)

        self._filters = filters
        self._project = project
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._cardinality = cardinality

    def build(self, input_shape):
        # identity
        if self._project == 'down':  # TODO: refactor to enum
            self._identity_conv = tf.layers.Conv2D(
                self._filters * 4, 3, 2, padding='same', use_bias=False,
                kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
            self._identity_bn = Normalization()

        elif self._project:
            self._identity_conv = tf.layers.Conv2D(
                self._filters * 4, 1, use_bias=False, kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer)
            self._identity_bn = Normalization()
        else:
            self._identity_conv = None
            self._identity_bn = None

        # conv_1
        self._conv_1 = tf.layers.Conv2D(
            self._filters * 2, 1, use_bias=False, kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer)
        self._bn_1 = Normalization()

        # conv_2
        self._conv_2 = []
        for _ in range(self._cardinality):
            strides = 2 if self._project == 'down' else 1
            conv = tf.layers.Conv2D(
                (self._filters * 2) // self._cardinality, 3, strides, padding='same', use_bias=False,
                kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
            self._conv_2.append(conv)

        self._bn_2 = []
        for _ in range(self._cardinality):
            bn = Normalization()
            self._bn_2.append(bn)

        # conv_3
        self._conv_3 = tf.layers.Conv2D(
            self._filters * 4, 1, use_bias=False, kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer)
        self._bn_3 = Normalization()

        super().build(input_shape)

    def call(self, input, training):
        # identity
        identity = input
        if self._identity_conv is not None:
            identity = self._identity_conv(identity)
        if self._identity_bn is not None:
            identity = self._identity_bn(identity, training=training)

        # conv_1
        input = self._conv_1(input)
        input = self._bn_1(input, training=training)
        input = tf.nn.relu(input)

        # conv_2
        splits = tf.split(input, len(self._conv_2), -1)
        transformations = []
        for split, conv, bn in zip(splits, self._conv_2, self._bn_2):
            split = conv(split)
            split = bn(split, training=training)
            split = tf.nn.relu(split)
            transformations.append(split)
        input = tf.concat(transformations, -1)

        # conv_3
        input = self._conv_3(input)
        input = self._bn_3(input, training=training)
        input = input + identity
        input = tf.nn.relu(input)

        return input


class ResNeXt_Block(tf.layers.Layer):
    def __init__(self, filters, depth, downsample, kernel_initializer, kernel_regularizer, name='resnext_block'):
        super().__init__(name=name)

        self._filters = filters
        self._depth = depth
        self._downsample = downsample
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        layers = []

        for i in range(self._depth):
            if i == 0:
                project = 'down' if self._downsample else True
            else:
                project = False

            layer = ResNeXt_Bottleneck(
                self._filters, project=project, kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer)
            layers.append(layer)

        self._layers = layers

    def call(self, input, training):
        for f in self._layers:
            input = f(input, training=training)

        return input


class ResNeXt_ConvInput(tf.layers.Layer):
    def __init__(self, kernel_initializer, kernel_regularizer, name='resnext_conv1'):
        super().__init__(name=name)

        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self._conv = tf.layers.Conv2D(
            64, 7, 2, padding='same', use_bias=False, kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer)
        self._bn = Normalization()

        super().build(input_shape)

    def call(self, input, training):
        input = self._conv(input)
        input = self._bn(input, training=training)
        input = tf.nn.relu(input)

        return input


class ResNeXt(tf.layers.Layer):
    def __init__(self, kernel_initializer, kernel_regularizer, name='resnext'):
        super().__init__(name=name)

        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer

    def call(self, input, training):
        input = self._conv_1(input, training=training)
        C1 = input
        input = self._conv_1_max_pool(input)
        input = self._conv_2(input, training=training)
        C2 = input
        input = self._conv_3(input, training=training)
        C3 = input
        input = self._conv_4(input, training=training)
        C4 = input
        input = self._conv_5(input, training=training)
        C5 = input

        return {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}


class ResNeXt_50(ResNeXt):
    # TODO: check activation is used
    def __init__(self, activation, kernel_initializer=None, kernel_regularizer=None, name='resnext_v2_50'):
        if kernel_initializer is None:
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)

        if kernel_regularizer is None:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        super().__init__(kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name)

    def build(self, input_shape):
        self._conv_1 = ResNeXt_ConvInput(
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self._conv_1_max_pool = tf.layers.MaxPooling2D(3, 2, padding='same')

        self._conv_2 = ResNeXt_Block(
            filters=64, depth=3, downsample=False, kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer)
        self._conv_3 = ResNeXt_Block(
            filters=128, depth=4, downsample=True, kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer)
        self._conv_4 = ResNeXt_Block(
            filters=256, depth=6, downsample=True, kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer)
        self._conv_5 = ResNeXt_Block(
            filters=512, depth=3, downsample=True, kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer)

        super().build(input_shape)


def main():
    image = tf.zeros((8, 224, 224, 3))

    net = ResNeXt_50()
    output = net(image, training=True)

    for k in output:
        shape = output[k].shape
        assert shape[1] == shape[2] == 224 // 2**int(k[1:]), 'invalid shape {} for layer {}'.format(shape, k)
        print(output[k])


if __name__ == '__main__':
    main()
