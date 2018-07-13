import tensorflow as tf
import utils
from utils import Detection


def focal_sigmoid_cross_entropy_with_logits(
        labels, logits, focus=2.0, alpha=0.25, eps=1e-7, name='focal_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        alpha = tf.fill(tf.shape(labels), alpha)
        prob = tf.nn.sigmoid(logits)
        prob_true = tf.where(tf.equal(labels, 1), prob, 1 - prob)
        alpha = tf.where(tf.equal(labels, 1), alpha, 1 - alpha)
        loss = -alpha * (1 - prob_true)**focus * tf.log(prob_true + eps)

        return loss


# TODO: check if this is correct
def focal_softmax_cross_entropy_with_logits(
        labels, logits, focus=2.0, alpha=0.25, eps=1e-7, name='focal_softmax_cross_entropy_with_logits'):
    with tf.name_scope(name):
        alpha = tf.ones_like(labels) * alpha

        prob = tf.nn.softmax(logits, -1)

        labels_eq_1 = tf.equal(labels, 1)
        a_balance = tf.where(labels_eq_1, alpha, 1 - alpha)
        prob_true = tf.where(labels_eq_1, prob, 1 - prob)
        modulating_factor = (1.0 - prob_true)**focus

        log_prob = tf.log(prob + eps)
        loss = -tf.reduce_sum(a_balance * modulating_factor * labels * log_prob, -1)

        return loss


# TODO: check bg mask usage and bg weighting calculation
def classification_loss(labels, logits, fg_mask, name='classification_loss'):
    with tf.name_scope(name):
        losses = []

        focal = focal_sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, alpha=0.9, focus=2)
        num_fg = tf.reduce_sum(tf.to_float(fg_mask))
        focal = tf.reduce_sum(focal) / tf.maximum(num_fg, 1.0)  # TODO: count all points, not only trainable?
        losses.append(focal)

        # bce = balanced_sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, axis=0)
        # losses.append(bce)
        #
        # dice = dice_loss(labels=labels, logits=logits, axis=0)
        # losses.append(dice)

        # jaccard = jaccard_loss(labels=labels, logits=logits, axis=0)
        # losses.append(jaccard)

        # iou = fixed_iou_loss(labels, logits, axis=0, smooth=1e-7)
        # losses.append(iou)

        # FIXME:
        # bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        # bce = tf.reshape(bce, [-1])
        # bce, _ = tf.nn.top_k(bce, 128, sorted=False)
        # tf.summary.histogram('loss', bce)  # FIXME:
        # losses.append(bce)

        loss = sum(tf.reduce_mean(l) for l in losses)

        return loss


def regression_loss(labels, logits, fg_mask, name='regression_loss'):
    with tf.name_scope(name):
        loss = tf.losses.huber_loss(
            labels=labels,
            predictions=logits,
            weights=tf.expand_dims(fg_mask, -1),
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return loss


def jaccard_loss(labels, logits, smooth=100., axis=None, name='jaccard_loss'):
    with tf.name_scope(name):
        logits = tf.nn.sigmoid(logits)

        intersection = tf.reduce_sum(labels * logits, axis=axis)
        union = tf.reduce_sum(labels, axis=axis) + tf.reduce_sum(logits, axis=axis)

        jaccard = (intersection + smooth) / (union - intersection + smooth)
        loss = (1 - jaccard) * smooth

        return loss


def dice_loss(labels, logits, smooth=1., axis=None, name='dice_loss'):
    with tf.name_scope(name):
        logits = tf.nn.sigmoid(logits)

        intersection = tf.reduce_sum(labels * logits, axis=axis)
        union = tf.reduce_sum(labels, axis=axis) + tf.reduce_sum(logits, axis=axis)

        coef = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - coef

        return loss


def fixed_iou_loss(labels, logits, smooth=1., axis=0, name='fixed_iou_loss'):
    with tf.name_scope(name):
        logits = tf.nn.sigmoid(logits)

        intersection = tf.reduce_sum(labels * logits, axis=axis)
        union = tf.reduce_sum(labels, axis=axis) + tf.reduce_sum((1 - labels) * logits, axis=axis)

        iou = (intersection + smooth) / (union + smooth)
        loss = 1 - iou

        return loss


# def balanced_sigmoid_cross_entropy_with_logits(
#         labels, logits, fg_mask, name='balanced_sigmoid_cross_entropy_with_logits'):
#     with tf.name_scope(name):
#         num_positive = tf.reduce_sum(tf.to_float(fg_mask))
#         num_negative = tf.reduce_sum(1. - tf.to_float(fg_mask))
#
#         weight_positive = num_negative / (num_positive + num_negative)
#         weight_negative = num_positive / (num_positive + num_negative)
#         ones = tf.ones_like(fg_mask, dtype=tf.float32)
#         weight = tf.where(fg_mask, ones * weight_positive, ones * weight_negative)
#         weight = tf.expand_dims(weight, -1)
#
#         loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
#         loss = loss * weight
#
#         return loss


def balanced_sigmoid_cross_entropy_with_logits(
        labels, logits, axis=None, name='balanced_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        num_positive = tf.reduce_sum(labels, axis=axis)
        num_negative = tf.reduce_sum(1 - labels, axis=axis)

        weight_positive = num_negative / (num_positive + num_negative)
        weight_negative = num_positive / (num_positive + num_negative)
        ones = tf.ones_like(labels)
        weight = tf.where(tf.equal(labels, 1), ones * weight_positive, ones * weight_negative)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = loss * weight

        return loss


def loss(labels: Detection, logits: Detection, name='loss'):
    with tf.name_scope(name):
        fg_mask = utils.classmap_decode(labels.classification.prob).fg_mask

        # FIXME:
        tf.summary.histogram(
            'prob_fg', tf.boolean_mask(logits.classification.prob, tf.equal(labels.classification.prob, 1)))
        tf.summary.histogram(
            'prob_bg', tf.boolean_mask(logits.classification.prob, tf.equal(labels.classification.prob, 0)))

        class_loss = classification_loss(
            labels=labels.classification.prob,
            logits=logits.classification.unscaled,
            fg_mask=fg_mask)

        regr_loss = regression_loss(
            labels=labels.regression,
            logits=logits.regression,
            fg_mask=fg_mask)

        return class_loss, regr_loss
