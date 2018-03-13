import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import math


def conv(input, filters, kernel_size, strides, kernel_initializer,
         bias_initializer, kernel_regularizer):
    return tf.layers.conv2d(
        input,
        filters,
        kernel_size,
        strides,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer)


def conv_norm_relu(input,
                   filters,
                   kernel_size,
                   strides,
                   dropout,
                   kernel_initializer,
                   bias_initializer,
                   kernel_regularizer,
                   norm_type,
                   training,
                   name='conv_norm_relu'):
    assert norm_type in ['layer', 'batch']

    with tf.name_scope(name):
        input = conv(
            input,
            filters,
            kernel_size,
            strides,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer)

        if norm_type == 'layer':
            input = tf.contrib.layers.layer_norm(input)
        elif norm_type == 'batch':
            input = tf.layers.batch_normalization(input, training=training)

        input = tf.nn.relu(input)
        input = tf.layers.dropout(input, rate=dropout, training=training)

        return input


def classification_subnet(input,
                          num_classes,
                          num_anchors,
                          dropout,
                          kernel_initializer,
                          bias_initializer,
                          kernel_regularizer,
                          norm_type,
                          training,
                          name='classification_subnet'):
    with tf.name_scope(name):
        for _ in range(4):
            input = conv_norm_relu(
                input,
                256,
                3,
                1,
                dropout=dropout,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                norm_type=norm_type,
                training=training)

        pi = 0.01
        bias_prior_initializer = tf.constant_initializer(
            -math.log((1 - pi) / pi))

        input = conv(
            input,
            num_anchors * num_classes,
            3,
            1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_prior_initializer,
            kernel_regularizer=kernel_regularizer)

        shape = tf.shape(input)
        input = tf.reshape(
            input, (shape[0], shape[1], shape[2], num_anchors, num_classes))

        return input


def regression_subnet(input,
                      num_anchors,
                      dropout,
                      kernel_initializer,
                      bias_initializer,
                      kernel_regularizer,
                      norm_type,
                      training,
                      name='regression_subnet'):
    with tf.name_scope(name):
        for _ in range(4):
            input = conv_norm_relu(
                input,
                256,
                3,
                1,
                dropout=dropout,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                norm_type=norm_type,
                training=training)

        input = conv(
            input,
            num_anchors * 4,
            3,
            1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer)

        sh = tf.shape(input)
        input = tf.reshape(input, (sh[0], sh[1], sh[2], num_anchors, 4))

        return input


def validate_level_shape(input, output, l, name='validate_level_shape'):
    with tf.name_scope(name):
        input_shape = tf.shape(input, out_type=tf.int32)
        output_shape = tf.shape(output, out_type=tf.int32)
        return tf.assert_equal(
            tf.to_float(output_shape[1:3]),
            tf.to_float(tf.ceil(input_shape[1:3] / 2**l)),
        )


def backbone(input, levels, training, name='backbone'):
    level_to_layer = [
        None,
        'resnet_v2_50/conv1',
        'resnet_v2_50/block1/unit_3/bottleneck_v2/conv1',
        'resnet_v2_50/block2/unit_4/bottleneck_v2/conv1',
        'resnet_v2_50/block3/unit_6/bottleneck_v2/conv1',
        'resnet_v2_50/block4',
    ]

    with tf.name_scope(name):
        with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
            _, outputs = nets.resnet_v2.resnet_v2_50(
                input,
                num_classes=None,
                global_pool=False,
                output_stride=None,
                is_training=training)

        bottom_up = []

        for l in levels:
            output = outputs[level_to_layer[l.number]]
            with tf.control_dependencies(
                [validate_level_shape(input, output, l.number)]):
                bottom_up.append(tf.identity(output))

        return bottom_up


def validate_lateral_shape(input, lateral, name='validate_lateral_shape'):
    with tf.name_scope(name):
        input_shape = tf.shape(input, out_type=tf.int32)
        lateral_shape = tf.shape(lateral, out_type=tf.int32)
        return tf.assert_equal(
            tf.to_float(tf.round(lateral_shape[1:3] / input_shape[1:3])),
            tf.to_float(2.0),
        )


def fpn(bottom_up,
        extra_levels,
        dropout,
        kernel_initializer,
        bias_initializer,
        kernel_regularizer,
        norm_type,
        training,
        name='fpn'):
    def conv(input, kernel_size, strides):
        return conv_norm_relu(
            input,
            256,
            kernel_size,
            strides,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            norm_type=norm_type,
            training=training)

    def upsample_merge(input, lateral):
        with tf.control_dependencies([validate_lateral_shape(input, lateral)]):
            lateral = conv(lateral, 1, 1)
            input = tf.image.resize_images(
                input,
                tf.shape(lateral)[1:3],
                method=tf.image.ResizeMethod.BILINEAR)

            return input + lateral

    with tf.name_scope(name):
        input = bottom_up[-1]
        top_down = []

        for l in extra_levels:
            input = conv(input, 3, 2)
            top_down.insert(0, input)

        input = conv(bottom_up.pop(), 1, 1)
        top_down.append(input)

        for _ in range(len(bottom_up)):
            input = upsample_merge(input, bottom_up.pop())
            input = conv(input, 3, 1)
            top_down.append(input)

        return top_down


def retinanet_base(input,
                   num_classes,
                   levels,
                   dropout,
                   kernel_initializer,
                   bias_initializer,
                   kernel_regularizer,
                   norm_type,
                   training,
                   name='retinanet_base'):
    backbone_levels = [l for l in levels if l.number <= 5]
    extra_levels = [l for l in levels if l.number > 5]

    with tf.name_scope(name):
        bottom_up = backbone(input, levels=backbone_levels, training=training)

        top_down = fpn(
            bottom_up,
            extra_levels=extra_levels,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            norm_type=norm_type,
            training=training)

        assert len(top_down) == len(levels)

        classifications = [
            classification_subnet(
                input,
                num_classes=num_classes,
                num_anchors=len(l.anchor_aspect_ratios) * len(
                    l.anchor_scale_ratios),
                dropout=dropout,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                norm_type=norm_type,
                training=training) for x, l in zip(reversed(top_down), levels)
        ]

        regressions = [
            regression_subnet(
                input,
                num_anchors=len(l.anchor_aspect_ratios) * len(
                    l.anchor_scale_ratios),
                dropout=dropout,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                norm_type=norm_type,
                training=training) for x, l in zip(reversed(top_down), levels)
        ]

        return classifications, regressions


def retinanet(input,
              num_classes,
              levels,
              dropout,
              weight_decay,
              norm_type,
              training,
              name='retinanet'):
    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    bias_initializer = tf.zeros_initializer()
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    return retinanet_base(
        input,
        num_classes=num_classes,
        levels=levels,
        dropout=dropout,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        norm_type=norm_type,
        training=training,
        name=name)
