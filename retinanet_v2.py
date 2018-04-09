import tensorflow as tf
import math
import utils
import resnet
import densenet
from network import Network, Sequential

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


def build_backbone(backbone, dropout_rate):
    assert backbone in ['resnet', 'densenet']
    if backbone == 'resnet':
        return resnet.ResNeXt_50()
    elif backbone == 'densenet':
        return densenet.DenseNetBC_169(dropout_rate=dropout_rate)


class ClassificationSubnet(Network):
    def __init__(self,
                 num_anchors,
                 num_classes,
                 kernel_initializer,
                 kernel_regularizer,
                 name='classification_subnet'):
        super().__init__(name=name)

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.pre_conv = self.track_layer(
            Sequential([
                Sequential([
                    tf.layers.Conv2D(
                        256,
                        3,
                        1,
                        padding='same',
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer),
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
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_initializer=bias_prior_initializer))

    def call(self, input, training):
        input = self.pre_conv(input, training)
        input = self.out_conv(input)

        shape = tf.shape(input)
        input = tf.reshape(
            input,
            (shape[0], shape[1], shape[2], self.num_anchors, self.num_classes))

        return input


class RegressionSubnet(Network):
    def __init__(self,
                 num_anchors,
                 kernel_initializer,
                 kernel_regularizer,
                 name='classification_subnet'):
        super().__init__(name=name)

        self.num_anchors = num_anchors

        self.pre_conv = self.track_layer(
            Sequential([
                Sequential([
                    tf.layers.Conv2D(
                        256,
                        3,
                        1,
                        padding='same',
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer),
                    tf.layers.BatchNormalization(),
                    tf.nn.relu,
                ]) for i in range(4)
            ]))

        self.out_conv = self.track_layer(
            tf.layers.Conv2D(
                num_anchors * 4,
                3,
                1,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer))

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


class FeaturePyramidNetwork(Network):
    class UpsampleMerge(Network):
        def __init__(self,
                     kernel_initializer,
                     kernel_regularizer,
                     name='upsample_merge'):
            super().__init__(name=name)

            self.conv_lateral = self.track_layer(
                Sequential([
                    tf.layers.Conv2D(
                        256,
                        1,
                        1,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer),
                    tf.layers.BatchNormalization()
                ]))

            self.conv_merge = self.track_layer(
                Sequential([
                    tf.layers.Conv2D(
                        256,
                        3,
                        1,
                        padding='same',
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer),
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
            merged = self.conv_merge(merged, training)

            return merged

    def __init__(self,
                 kernel_initializer,
                 kernel_regularizer,
                 name='feature_pyramid_network'):
        super().__init__(name=name)

        self.p6_from_c5 = self.track_layer(
            Sequential([
                tf.layers.Conv2D(
                    256,
                    3,
                    2,
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                tf.layers.BatchNormalization()
            ]))

        self.p7_from_p6 = self.track_layer(
            Sequential([
                tf.nn.relu,
                tf.layers.Conv2D(
                    256,
                    3,
                    2,
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                tf.layers.BatchNormalization()
            ]))

        self.p5_from_c5 = self.track_layer(
            Sequential([
                tf.layers.Conv2D(
                    256,
                    1,
                    1,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                tf.layers.BatchNormalization()
            ]))

        self.p4_from_c4p5 = self.track_layer(
            FeaturePyramidNetwork.UpsampleMerge(
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='upsample_merge_c4p5'))
        self.p3_from_c3p4 = self.track_layer(
            FeaturePyramidNetwork.UpsampleMerge(
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='upsample_merge_c3p4'))

    def call(self, input, training):
        P6 = self.p6_from_c5(input['C5'], training)
        P7 = self.p7_from_p6(P6, training)
        P5 = self.p5_from_c5(input['C5'], training)
        P4 = self.p4_from_c4p5(input['C4'], P5, training)
        P3 = self.p3_from_c3p4(input['C3'], P4, training)

        return {'P3': P3, 'P4': P4, 'P5': P5, 'P6': P6, 'P7': P7}


class RetinaNetBase(Network):
    def __init__(self,
                 backbone,
                 levels,
                 num_classes,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='retinanet_base'):
        super().__init__(name=name)

        self.backbone = self.track_layer(
            build_backbone(backbone, dropout_rate=dropout_rate))

        if backbone == 'densenet':
            # DenseNet has preactivation architecture,
            # so we need to apply acitvation before passing features to FPN
            self.postprocess_bottom_up = {
                cn: self.track_layer(
                    Sequential([
                        tf.layers.BatchNormalization(),
                        tf.nn.relu,
                    ]))
                for cn in ['C3', 'C4', 'C5']
            }
        else:
            self.postprocess_bottom_up = None

        self.fpn = self.track_layer(
            FeaturePyramidNetwork(
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer))

        # TODO: level anchor boxes
        self.classification_subnet = self.track_layer(
            ClassificationSubnet(
                num_anchors=levels['P3'].anchor_boxes.shape[0],
                num_classes=num_classes,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='classification_subnet'))

        # TODO: level anchor boxes
        self.regression_subnet = self.track_layer(
            RegressionSubnet(
                num_anchors=levels['P3'].anchor_boxes.shape[0],
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='regression_subnet'))

    def call(self, input, training):
        bottom_up = self.backbone(input, training)

        if self.postprocess_bottom_up is not None:
            bottom_up = {
                cn: self.postprocess_bottom_up[cn](bottom_up[cn], training)
                for cn in ['C3', 'C4', 'C5']
            }

        top_down = self.fpn(bottom_up, training)

        classifications = {
            k: self.classification_subnet(top_down[k], training)
            for k in top_down
        }

        regressions = {
            k: self.regression_subnet(top_down[k], training)
            for k in top_down
        }

        return classifications, regressions


class RetinaNet(Network):
    def __init__(self,
                 backbone,
                 levels,
                 num_classes,
                 dropout_rate,
                 name='retinanet'):
        super().__init__(name=name)

        self.levels = levels

        kernel_initializer = tf.random_normal_initializer(
            mean=0.0, stddev=0.01)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        self.base = RetinaNetBase(
            backbone=backbone,
            levels=levels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

    def call(self, input, training):
        image_size = tf.shape(input)[1:3]
        classifications, regressions = self.base(input, training)

        regressions = {
            pn: regression_postprocess(
                regressions[pn],
                tf.to_float(self.levels[pn].anchor_boxes / image_size))
            for pn in self.levels
        }

        return classifications, regressions


def scale_regression(regression, anchor_boxes):
    anchor_boxes = tf.tile(anchor_boxes, (1, 2))
    anchor_boxes = tf.reshape(
        anchor_boxes, (1, 1, 1, anchor_boxes.shape[0], anchor_boxes.shape[1]))

    return regression * anchor_boxes


# TODO: make suitable for tf.eager
def regression_postprocess(regression,
                           anchor_boxes,
                           name='regression_postprocess'):
    with tf.name_scope(name):
        regression = scale_regression(regression, anchor_boxes)
        regression = utils.boxmap_anchor_relative_to_image_relative(regression)
        regression = utils.boxmap_center_relative_to_corner_relative(
            regression)

        return regression
