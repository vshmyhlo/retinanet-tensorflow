import tensorflow as tf
from network import Network, Sequential
import functools
import tensorflow.contrib.slim as slim


# TODO: remove `track_layer(...)` stuff
# TODO: initialization
# TODO: reegularization
# TODO: batchnorm
# TODO: activation parameter
# TODO: dropout
# TODO: private fields
# TODO: check shapes
# TODO: move everything to `build`
def expanded_conv(input_tensor,
                  num_outputs,
                  expansion_size,
                  stride=1,
                  kernel_size=(3, 3),
                  residual=True,
                  normalizer_fn=None,
                  expansion_transform=None,
                  depthwise_location='expansion',
                  endpoints=None,
                  use_explicit_padding=False,
                  padding='SAME',
                  scope=None):
    """Depthwise Convolution Block with expansion.
    Builds a composite convolution that has the following structure
    expansion (1x1) -> depthwise (kernel_size) -> projection (1x1)
    Args:
      input_tensor: input
      num_outputs: number of outputs in the final layer.
      expansion_size: the size of expansion, could be a constant or a callable.
        If latter it will be provided 'num_inputs' as an input. For forward
        compatibility it should accept arbitrary keyword arguments.
        Default will expand the input by factor of 6.
      stride: depthwise stride
      rate: depthwise rate
      kernel_size: depthwise kernel
      residual: whether to include residual connection between input
        and output.
      normalizer_fn: batchnorm or otherwise
      split_projection: how many ways to split projection operator
        (that is conv expansion->bottleneck)
      split_expansion: how many ways to split expansion op
        (that is conv bottleneck->expansion) ops will keep depth divisible
        by this value.
      expansion_transform: Optional function that takes expansion
        as a single input and returns output.
      depthwise_location: where to put depthwise covnvolutions supported
        values None, 'input', 'output', 'expansion'
      depthwise_channel_multiplier: depthwise channel multiplier:
      each input will replicated (with different filters)
      that many times. So if input had c channels,
      output will have c x depthwise_channel_multpilier.
      endpoints: An optional dictionary into which intermediate endpoints are
        placed. The keys "expansion_output", "depthwise_output",
        "projection_output" and "expansion_transform" are always populated, even
        if the corresponding functions are not invoked.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      padding: Padding type to use if `use_explicit_padding` is not set.
      scope: optional scope.
    Returns:
      Tensor of depth num_outputs
    Raises:
      TypeError: on inval
    """
    prev_depth = input_tensor.get_shape().as_list()[3]
    depthwise_func = functools.partial(
        slim.separable_conv2d,
        num_outputs=None,
        kernel_size=kernel_size,
        stride=stride,
        normalizer_fn=normalizer_fn,
        padding=padding,
        scope='depthwise')
    # b1 -> b2 * r -> b2
    #   i -> (o * r) (bottleneck) -> o
    net = input_tensor

    inner_size = expansion_size

    net = split_conv(
        net,
        inner_size,
        scope='expand',
        stride=1,
        normalizer_fn=normalizer_fn)
    net = tf.identity(net, 'expansion_output')

    if depthwise_location == 'expansion':
        net = depthwise_func(net)

    net = tf.identity(net, name='depthwise_output')
    # Note in contrast with expansion, we always have
    # projection to produce the desired output size.
    net = split_conv(
        net,
        num_outputs,
        stride=1,
        scope='project',
        normalizer_fn=normalizer_fn,
        activation_fn=tf.identity)

    if (residual and
            # stride check enforces that we don't add residuals when spatial
            # dimensions are None
            stride == 1 and
            # Depth matches
            net.get_shape().as_list()[3] ==
            input_tensor.get_shape().as_list()[3]):
        net += input_tensor
    return tf.identity(net, name='output')


# input = tf.zeros((1, 224, 224, 3))
# expanded_conv(input, stride=2, num_outputs=24, expansion_size=6, residual=True, normalizer_fn=slim.batch_norm)
# fail

class DepthwiseConv2D(Network):
    def __init__(self, kernel_size, strides, padding, use_bias, kernel_initializer, kernel_regularizer,
                 name='separable_conv2d'):
        super().__init__(name=name)

        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self._kernel = self.add_variable(
            'kernel', (self._kernel_size, self._kernel_size, input_shape[3], 1), initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer)

        super().build(input_shape)

    def call(self, input):
        input = tf.nn.depthwise_conv2d(
            input, self._kernel, strides=[1, self._strides, self._strides, 1], padding=self._padding.upper())

        return input


class Bottleneck(Network):
    def __init__(self, filters, strides, expansion_factor, activation, dropout_rate, kernel_initializer,
                 kernel_regularizer,
                 name='bottleneck'):
        super().__init__(name=name)

        self._filters = filters
        self._strides = strides
        self._expansion_factor = expansion_factor
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.expand_conv = Sequential([
            tf.layers.Conv2D(
                input_shape[3] * self._expansion_factor, 1, use_bias=False, kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer),
            tf.layers.BatchNormalization(),
            self._activation,
            tf.layers.Dropout(self._dropout_rate)
        ])

        self.depthwise_conv = Sequential([
            DepthwiseConv2D(
                3, strides=self._strides, padding='same', use_bias=False, kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer),
            tf.layers.BatchNormalization(),
            self._activation,
            tf.layers.Dropout(self._dropout_rate)
        ])

        self.linear_conv = Sequential([
            tf.layers.Conv2D(
                self._filters, 1, use_bias=False, kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer),
            tf.layers.BatchNormalization(),
            tf.layers.Dropout(self._dropout_rate)
        ])

        super().build(input_shape)

    def call(self, input, training):
        identity = input

        input = self.expand_conv(input, training)
        input = self.depthwise_conv(input, training)
        input = self.linear_conv(input, training)

        if input.shape == identity.shape:
            input = input + identity

        return input


class MobileNetV2(Network):
    def __init__(self, activation, dropout_rate, name='mobilenet_v2'):
        super().__init__(name=name)

        if activation is None:
            self._activation = tf.nn.relu6
        else:
            self._activation = activation
        self._dropout_rate = dropout_rate
        self._kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        self._kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=4e-5)

    def build(self, input_shape):
        self.input_conv = Sequential([
            tf.layers.Conv2D(
                32, 3, strides=2, padding='same', use_bias=False, kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer),
            tf.layers.BatchNormalization(),
            self._activation,
            tf.layers.Dropout(self._dropout_rate)
        ])

        self.bottleneck_1_1 = Bottleneck(
            16, expansion_factor=1, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)

        self.bottleneck_2_1 = Bottleneck(
            24, expansion_factor=6, strides=2, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.bottleneck_2_2 = Bottleneck(
            24, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)

        self.bottleneck_3_1 = Bottleneck(
            32, expansion_factor=6, strides=2, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.bottleneck_3_2 = Bottleneck(
            32, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.bottleneck_3_3 = Bottleneck(
            32, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)

        self.bottleneck_4_1 = Bottleneck(
            64, expansion_factor=6, strides=2, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.bottleneck_4_2 = Bottleneck(
            64, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.bottleneck_4_3 = Bottleneck(
            64, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.bottleneck_4_4 = Bottleneck(
            64, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)

        self.bottleneck_5_1 = Bottleneck(
            96, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.bottleneck_5_2 = Bottleneck(
            96, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.bottleneck_5_3 = Bottleneck(
            96, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)

        self.bottleneck_6_1 = Bottleneck(
            160, expansion_factor=6, strides=2, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.bottleneck_6_2 = Bottleneck(
            160, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)
        self.bottleneck_6_3 = Bottleneck(
            160, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)

        self.bottleneck_7_1 = Bottleneck(
            320, expansion_factor=6, strides=1, activation=self._activation, dropout_rate=self._dropout_rate,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)

        self.output_conv = Sequential([
            tf.layers.Conv2D(
                32, 1, use_bias=False, kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer),
            tf.layers.BatchNormalization(),
            self._activation,
            tf.layers.Dropout(self._dropout_rate)
        ])

        super().build(input_shape)

    def call(self, input, training):
        input = self.input_conv(input, training)

        input = self.bottleneck_1_1(input, training)
        C1 = input

        input = self.bottleneck_2_1(input, training)
        input = self.bottleneck_2_2(input, training)
        C2 = input

        input = self.bottleneck_3_1(input, training)
        input = self.bottleneck_3_2(input, training)
        input = self.bottleneck_3_3(input, training)
        C3 = input

        input = self.bottleneck_4_1(input, training)
        input = self.bottleneck_4_2(input, training)
        input = self.bottleneck_4_3(input, training)
        input = self.bottleneck_4_4(input, training)

        input = self.bottleneck_5_1(input, training)
        input = self.bottleneck_5_2(input, training)
        input = self.bottleneck_5_3(input, training)
        C4 = input

        input = self.bottleneck_6_1(input, training)
        input = self.bottleneck_6_2(input, training)
        input = self.bottleneck_6_3(input, training)

        input = self.bottleneck_7_1(input, training)

        input = self.output_conv(input, training)
        C5 = input

        return {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}


if __name__ == '__main__':
    input = tf.zeros((1, 224, 224, 3))
    net = MobileNetV2(activation=tf.nn.elu, dropout_rate=0.2)
    output = net(input, training=True)

    for k in output:
        assert output[k].shape[1] == output[k].shape[2] == 224 / 2**int(k[1:]), 'invalid shape {} for layer {}'.format(
            output[k].shape, k)
