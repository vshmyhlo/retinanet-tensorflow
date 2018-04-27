import tensorflow as tf
import math
import resnet
import densenet
from network import Network, Sequential


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
                        use_bias=False,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer),
                    tf.layers.BatchNormalization(),
                    tf.nn.relu,
                ]) for _ in range(4)
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
        input = tf.reshape(input, (shape[0], shape[1], shape[2], self.num_anchors, self.num_classes))

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
                        use_bias=False,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer),
                    tf.layers.BatchNormalization(),
                    tf.nn.relu,
                ]) for _ in range(4)
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
        input = tf.reshape(input, (shape[0], shape[1], shape[2], self.num_anchors, 4))

        return input


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
                        use_bias=False,
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
                        use_bias=False,
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
                    use_bias=False,
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
                    use_bias=False,
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
                    use_bias=False,
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
            # so we need to apply activation before passing features to FPN
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

        # all pyramid levels must have the same number of anchors
        num_anchors = set(levels[pn].anchor_sizes.shape[0] for pn in levels)
        assert len(num_anchors) == 1
        num_anchors = list(num_anchors)[0]

        self.classification_subnet = self.track_layer(
            ClassificationSubnet(
                num_anchors=num_anchors,  # TODO: level anchor boxes
                num_classes=num_classes,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='classification_subnet'))

        self.regression_subnet = self.track_layer(
            RegressionSubnet(
                num_anchors=num_anchors,  # TODO: level anchor boxes
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
    def __init__(self, backbone, levels, num_classes, dropout_rate, name='retinanet'):
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
        return self.base(input, training)
