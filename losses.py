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
def classification_loss(labels, logits, non_bg_mask, class_loss_kwargs):
    losses = []

    focal = focal_sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, **class_loss_kwargs)
    num_non_bg = tf.reduce_sum(tf.to_float(non_bg_mask))
    focal = tf.reduce_sum(focal) / tf.maximum(num_non_bg, 1.0)  # TODO: count all points, not only trainable?
    losses.append(focal)

    # bce = balanced_sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, non_bg_mask=non_bg_mask)
    # losses.append(bce)
    #
    # dice = dice_loss(labels=labels, logits=logits, axis=0)
    # losses.append(dice)

    # jaccard = jaccard_loss(labels=labels, logits=logits, axis=0)
    # losses.append(jaccard)

    loss = sum(tf.reduce_mean(l) for l in losses)

    return loss


def regression_loss(labels, logits, non_bg_mask):
    loss = tf.losses.huber_loss(
        labels=labels,
        predictions=logits,
        weights=tf.expand_dims(non_bg_mask, -1),
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    check = tf.Assert(tf.is_finite(loss), [tf.reduce_mean(loss)])
    with tf.control_dependencies([check]):
        loss = tf.identity(loss)

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


def balanced_sigmoid_cross_entropy_with_logits(
        labels, logits, non_bg_mask, name='balanced_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        num_positive = tf.reduce_sum(tf.to_float(non_bg_mask))
        num_negative = tf.reduce_sum(1. - tf.to_float(non_bg_mask))

        weight_positive = num_negative / (num_positive + num_negative)
        weight_negative = num_positive / (num_positive + num_negative)
        ones = tf.ones_like(logits)
        weight = tf.where(tf.equal(labels, 1), ones * weight_positive, ones * weight_negative)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = loss * weight

        return loss


def classwise_balanced_sigmoid_cross_entropy_with_logits(
        labels, logits, axis=None, name='balanced_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        num_positive = tf.reduce_sum(tf.to_float(tf.equal(labels, 1)), axis=axis)
        num_negative = tf.reduce_sum(tf.to_float(tf.equal(labels, 0)), axis=axis)

        weight_positive = num_negative / (num_positive + num_negative)
        weight_negative = num_positive / (num_positive + num_negative)
        ones = tf.ones_like(logits)
        weight = tf.where(tf.equal(labels, 1), ones * weight_positive, ones * weight_negative)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = loss * weight

        return loss


def loss(labels: Detection, logits: Detection, class_loss_kwargs, name='loss'):
    with tf.name_scope(name):
        non_bg_mask = utils.classmap_decode(labels.classification.prob)['non_bg_mask']

        tf.summary.histogram('fg', tf.boolean_mask(logits.classification.prob, non_bg_mask))
        tf.summary.histogram('bg', tf.boolean_mask(logits.classification.prob, tf.logical_not(non_bg_mask)))

        class_loss = classification_loss(
            labels=labels.classification.prob,
            logits=logits.classification.unscaled,
            non_bg_mask=non_bg_mask,
            class_loss_kwargs=class_loss_kwargs)

        regr_loss = regression_loss(
            labels=labels.regression,
            logits=logits.regression,
            non_bg_mask=non_bg_mask)

        return class_loss, regr_loss
