import tensorflow as tf


def focal_sigmoid_cross_entropy_with_logits(labels, logits, focus=2.0, alpha=0.25,
                                            name='focal_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        alpha = tf.ones_like(labels) * alpha
        labels_eq_1 = tf.equal(labels, 1)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        prob = tf.nn.sigmoid(logits)
        a_balance = tf.where(labels_eq_1, alpha, 1 - alpha)
        prob_true = tf.where(labels_eq_1, prob, 1 - prob)
        modulating_factor = (1.0 - prob_true)**focus

        return a_balance * modulating_factor * loss


# TODO: check this is corrrect
def focal_softmax_cross_entropy_with_logits(labels, logits, focus=2.0, alpha=0.25,
                                            name='focal_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        alpha = tf.ones_like(labels) * alpha

        prob = tf.nn.softmax(logits, -1)

        labels_eq_1 = tf.equal(labels, 1)
        a_balance = tf.where(labels_eq_1, alpha, 1 - alpha)
        prob_true = tf.where(labels_eq_1, prob, 1 - prob)
        modulating_factor = (1.0 - prob_true)**focus

        log_prob = tf.log(prob)
        loss = -tf.reduce_sum(a_balance * modulating_factor * labels * log_prob, -1)

        return loss


def safe_div(numerator, denominator):
    return tf.where(tf.greater(denominator, 0),
                    tf.div(numerator, tf.where(tf.equal(denominator, 0), tf.ones_like(denominator), denominator)),
                    tf.zeros_like(numerator))


def classification_loss(labels, logits, non_background_mask):
    # TODO: choose loss
    class_loss = focal_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    class_loss = safe_div(tf.reduce_sum(class_loss), tf.reduce_sum(tf.to_float(non_background_mask)))

    return class_loss


def regression_loss(labels, logits, non_background_mask):
    regr_loss = tf.losses.huber_loss(
        labels=labels,
        predictions=logits,
        weights=tf.expand_dims(non_background_mask, -1),
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    return regr_loss


def merge_outputs(tensors, ignored_mask, name='merge_outputs'):
    with tf.name_scope(name):
        res = []
        for pn in tensors:
            mask = ignored_mask[pn]
            tensor = tensors[pn]
            tensor = tf.boolean_mask(tensor, mask)
            res.append(tensor)

        return tf.concat(res, 0)


def loss(labels, logits, not_ignored_mask, name='loss'):
    with tf.name_scope(name):
        labels = tuple(merge_outputs(x, not_ignored_mask) for x in labels)
        logits = tuple(merge_outputs(x, not_ignored_mask) for x in logits)

        class_labels, regr_labels = labels
        class_logits, regr_logits = logits

        non_background_mask = tf.not_equal(tf.argmax(class_labels, -1), 0)

        class_loss = classification_loss(
            labels=class_labels,
            logits=class_logits,
            non_background_mask=non_background_mask)
        regr_loss = regression_loss(
            labels=regr_labels,
            logits=regr_logits,
            non_background_mask=non_background_mask)

        return class_loss, regr_loss
