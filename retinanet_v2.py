import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import math
import utils
import tensorflow.contrib.eager as tfe
import resnet

# def conv_norm_relu(input,
#                    filters,
#                    kernel_size,
#                    strides,
#                    dropout,
#                    kernel_initializer,
#                    bias_initializer,
#                    kernel_regularizer,
#                    norm_type,
#                    training,
#                    name='conv_norm_relu'):
#     assert norm_type in ['layer', 'batch']
#
#     with tf.name_scope(name):
#         input = conv(
#             input,
#             filters,
#             kernel_size,
#             strides,
#             kernel_initializer=kernel_initializer,
#             bias_initializer=bias_initializer,
#             kernel_regularizer=kernel_regularizer)
#
#         if norm_type == 'layer':
#             input = tf.contrib.layers.layer_norm(input)
#         elif norm_type == 'batch':
#             input = tf.layers.batch_normalization(input, training=training)
#
#         input = tf.nn.relu(input)
#         shape = tf.shape(input)
#         input = tf.layers.dropout(
#             input,
#             rate=dropout,
#             noise_shape=(shape[0], 1, 1, shape[3]),
#             training=training)
#
#         return input


class ClassificationSubnet(tfe.Network):
    def __init__(self, num_anchors, num_classes, name='classification_subnet'):
        super().__init__(name=name)

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.pre_conv = self.track_layer(
            tfe.Sequential([
                tfe.Sequential([
                    tf.layers.Conv2D(256, 3, 1, padding='same'),
                    tf.layers.BatchNormalization(),
                    tf.nn.relu,
                ]) for i in range(4)
            ]))

        pi = 0.01
        bias_prior_initializer = tf.constant_initializer(
            -math.log((1 - pi) / pi))

        self.out_conv = self.track_layer(
            tf.layers.Conv2D(
                num_anchors * num_classes,
                3,
                1,
                padding='same',
                bias_initializer=bias_prior_initializer))

    def call(self, input, training):
        input = self.pre_conv(input, training)
        input = self.out_conv(input)

        shape = tf.shape(input)
        input = tf.reshape(
            input,
            (shape[0], shape[1], shape[2], self.num_anchors, self.num_classes))

        return input


class RegressionSubnet(tfe.Network):
    def __init__(self, num_anchors, name='classification_subnet'):
        super().__init__(name=name)

        self.num_anchors = num_anchors

        self.pre_conv = self.track_layer(
            tfe.Sequential([
                tfe.Sequential([
                    tf.layers.Conv2D(256, 3, 1, padding='same'),
                    tf.layers.BatchNormalization(),
                    tf.nn.relu,
                ]) for i in range(4)
            ]))

        self.out_conv = self.track_layer(
            tf.layers.Conv2D(num_anchors * 4, 3, 1, padding='same'))

    def call(self, input, training):
        input = self.pre_conv(input, training)
        input = self.out_conv(input)

        shape = tf.shape(input)
        input = tf.reshape(input,
                           (shape[0], shape[1], shape[2], self.num_anchors, 4))

        shifts, scales = tf.split(input, 2, -1)
        scales = tf.exp(scales)
        input = tf.concat([shifts, scales], -1)

        return input


# def validate_level_shape(input, output, l, name='validate_level_shape'):
#     with tf.name_scope(name):
#         input_size = tf.shape(input)[1:3]
#         output_size = tf.shape(output)[1:3]
#
#         return tf.assert_equal(
#             tf.to_int32(output_size), tf.to_int32(tf.ceil(input_size / 2**l)))

# def backbone(input, levels, training, name='backbone'):
#     level_to_layer = [
#         None,
#         'resnet_v2_50/conv1',
#         'resnet_v2_50/block1/unit_3/bottleneck_v2/conv1',
#         'resnet_v2_50/block2/unit_4/bottleneck_v2/conv1',
#         'resnet_v2_50/block3/unit_6/bottleneck_v2/conv1',
#         'resnet_v2_50/block4',
#     ]
#
#     with tf.name_scope(name):
#         with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
#             _, outputs = nets.resnet_v2.resnet_v2_50(
#                 input,
#                 num_classes=None,
#                 global_pool=False,
#                 output_stride=None,
#                 is_training=training)
#
#         bottom_up = []
#         validations = []
#
#         for l in levels:
#             output = outputs[level_to_layer[l.number]]
#             bottom_up.append(output)
#             validations.append(validate_level_shape(input, output, l.number))
#
#         with tf.control_dependencies(validations):
#             bottom_up = [tf.identity(x) for x in bottom_up]
#
#         return bottom_up

# def validate_lateral_shape(input, lateral, name='validate_lateral_shape'):
#     with tf.name_scope(name):
#         input_size = tf.shape(input)[1:3]
#         lateral_size = tf.shape(lateral)[1:3]
#         return tf.assert_equal(
#             tf.to_int32(tf.round(lateral_size / input_size)), 2)


class FeaturePyramidNetwork(tfe.Network):
    class UpsampleMerge(tfe.Network):
        def __init__(self, name='upsample_merge'):
            super().__init__(name=name)

            self.conv_lateral = self.track_layer(
                tfe.Sequential([
                    tf.layers.Conv2D(256, 1, 1),
                    tf.layers.BatchNormalization()
                ]))

            self.conv_merge = self.track_layer(
                tfe.Sequential([
                    tf.layers.Conv2D(256, 3, 1, padding='same'),
                    tf.layers.BatchNormalization()
                ]))

        def call(self, lateral, downsampled, training):
            lateral = self.conv_lateral(lateral, training)
            lateral_size = tf.shape(lateral)[1:3]
            downsampled = tf.image.resize_images(
                downsampled,
                lateral_size,
                method=tf.image.ResizeMethod.BILINEAR)

            merged = lateral + downsampled
            merged = self.conv_merge(merged)

            return merged

    def __init__(self, name='feature_pyramid_network'):
        super().__init__(name=name)

        self.p6_from_c5 = self.track_layer(
            tfe.Sequential([
                tf.layers.Conv2D(256, 3, 2, padding='same'),
                tf.layers.BatchNormalization()
            ]))

        self.p7_from_p6 = self.track_layer(
            tfe.Sequential([
                tf.nn.relu,
                tf.layers.Conv2D(256, 3, 2, padding='same'),
                tf.layers.BatchNormalization()
            ]))

        self.p5_from_c5 = self.track_layer(
            tfe.Sequential(
                [tf.layers.Conv2D(256, 1, 1),
                 tf.layers.BatchNormalization()]))

        self.p4_from_c4p5 = self.track_layer(
            FeaturePyramidNetwork.UpsampleMerge(name='upsample_merge_c4p5'))
        self.p3_from_c3p4 = self.track_layer(
            FeaturePyramidNetwork.UpsampleMerge(name='upsample_merge_c3p4'))

    def call(self, input, training):
        P6 = self.p6_from_c5(input['C5'], training)
        P7 = self.p7_from_p6(P6, training)
        P5 = self.p5_from_c5(input['C5'], training)
        P4 = self.p4_from_c4p5(input['C4'], P5, training)
        P3 = self.p3_from_c3p4(input['C3'], P4, training)

        return {'P3': P3, 'P4': P4, 'P5': P5, 'P6': P6, 'P7': P7}


class RetinaNetBase(tfe.Network):
    def __init__(self, levels, num_classes, name='retinanet_base'):
        super().__init__(name=name)

        self.backbone = self.track_layer(resnet.ResNeXt_50())
        self.fpn = self.track_layer(FeaturePyramidNetwork())

        self.classification_subnets = {
            pn: self.track_layer(
                ClassificationSubnet(
                    num_anchors=levels[pn].anchor_boxes.shape[0],
                    num_classes=num_classes,
                    name='classification_subnet_{}'.format(pn)))
            for pn in ['P3', 'P4', 'P5', 'P6', 'P7']
        }

        self.regression_subnets = {
            pn: self.track_layer(
                RegressionSubnet(
                    num_anchors=levels[pn].anchor_boxes.shape[0],
                    name='regression_subnet_{}'.format(pn)))
            for pn in ['P3', 'P4', 'P5', 'P6', 'P7']
        }

    def call(self, input, training):
        bottom_up = self.backbone(input, training)
        top_down = self.fpn(bottom_up, training)

        classifications = {
            k: self.classification_subnets[k](top_down[k], training)
            for k in top_down
        }

        regressions = {
            k: self.regression_subnets[k](top_down[k], training)
            for k in top_down
        }

        return classifications, regressions


class RetinaNet(tfe.Network):
    def __init__(self, levels, num_classes, name='retinanet'):
        super().__init__(name=name)

        self.base = RetinaNetBase(levels=levels, num_classes=num_classes)

    def call(self, input, training):
        classifications, regressions = self.base(input, training)

        return classifications, regressions


# def retinanet(input,
#               num_classes,
#               levels,
#               dropout,
#               weight_decay,
#               norm_type,
#               training,
#               name='retinanet'):
#     image_size = tf.shape(input)[1:3]
#     kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
#     bias_initializer = tf.zeros_initializer()
#     kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
#
#     classifications, regressions = retinanet_base(
#         input,
#         num_classes=num_classes,
#         levels=levels,
#         dropout=dropout,
#         kernel_initializer=kernel_initializer,
#         bias_initializer=bias_initializer,
#         kernel_regularizer=kernel_regularizer,
#         norm_type=norm_type,
#         training=training,
#         name=name)
#
#     regressions = tuple(
#         regression_postprocess(r, tf.to_float(l.anchor_boxes / image_size))
#         for r, l in zip(regressions, levels))
#
#     return classifications, regressions


def scale_regression(regression, anchor_boxes):
    anchor_boxes = tf.tile(anchor_boxes, (1, 2))
    anchor_boxes = tf.reshape(
        anchor_boxes, (1, 1, 1, anchor_boxes.shape[0], anchor_boxes.shape[1]))

    return regression * anchor_boxes


def regression_postprocess(regression,
                           anchor_boxes,
                           name='regression_postprocess'):
    with tf.name_scope(name):
        regression = scale_regression(regression, anchor_boxes)
        regression = utils.boxmap_anchor_relative_to_image_relative(regression)
        regression = utils.boxmap_center_relative_to_corner_relative(
            regression)

        return regression
