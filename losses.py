import tensorflow as tf
import utils


def focal_sigmoid_cross_entropy_with_logits(
        labels, logits, focus=2.0, alpha=0.25, name='focal_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        alpha = tf.ones_like(labels) * alpha
        labels_eq_1 = tf.equal(labels, 1)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        prob = tf.nn.sigmoid(logits)
        a_balance = tf.where(labels_eq_1, alpha, 1 - alpha)
        prob_true = tf.where(labels_eq_1, prob, 1 - prob)
        modulating_factor = (1.0 - prob_true)**focus

        return a_balance * modulating_factor * loss


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


# def classification_loss(labels, logits, non_background_mask):
#     num_non_background = tf.reduce_sum(tf.to_float(non_background_mask))
#     class_loss = focal_sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
#     class_loss = tf.reduce_sum(class_loss) / tf.maximum(num_non_background, 1.0)
#
#     return class_loss

def classification_loss(labels, logits, non_bg_mask):
    # TODO: check bg mask usage and bg weighting calculation

    loss = sum([
        tf.reduce_mean(balanced_sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)),
        tf.reduce_mean(dice_loss(labels=labels, logits=logits)),
    ])

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


def dice_loss(labels, logits, smooth=1, name='dice_loss'):
    with tf.name_scope(name):
        probs = tf.nn.sigmoid(logits)

        intersection = tf.reduce_sum(labels * probs)
        union = tf.reduce_sum(labels, -1) + tf.reduce_sum(probs)

        coef = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - coef

        return loss


def balanced_sigmoid_cross_entropy_with_logits(labels, logits, name='balanced_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        num_positive = tf.reduce_sum(tf.to_float(tf.equal(labels, 1)))
        num_negative = tf.reduce_sum(tf.to_float(tf.equal(labels, 0)))

        weight_positive = num_negative / (num_positive + num_negative)
        weight_negative = num_positive / (num_positive + num_negative)
        ones = tf.ones_like(logits)
        weight = tf.where(tf.equal(labels, 1), ones * weight_positive, ones * weight_negative)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = loss * weight

        return loss


def loss(labels, logits, name='loss'):
    with tf.name_scope(name):
        non_bg_mask = utils.classmap_decode(labels['classifications'])['non_bg_mask']

        class_loss = classification_loss(
            labels=labels['classifications'],
            logits=logits['classifications'],
            non_bg_mask=non_bg_mask)
        regr_loss = regression_loss(
            labels=labels['regressions'],
            logits=logits['regressions'],
            non_bg_mask=non_bg_mask)

        return class_loss, regr_loss
