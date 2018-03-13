import tensorflow as tf
import utils


def focal_sigmoid_cross_entropy_with_logits(
        labels,
        logits,
        focus=2.0,
        alpha=0.25,
        name='focal_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)

        a_balance = alpha * labels + (1 - alpha) * (1 - labels)

        prob = tf.nn.sigmoid(logits)
        prob_true = prob * labels + (1 - prob) * (1 - labels)
        modulating_factor = (1.0 - prob_true)**focus

        return a_balance * modulating_factor * loss


def loss(labels, logits, name='loss'):
    with tf.name_scope(name):
        assert len(labels[0]) == len(labels[1]) == len(logits[0]) == len(
            logits[1])

        labels = tuple(utils.merge_outputs(x) for x in labels)
        logits = tuple(utils.merge_outputs(x) for x in logits)

        non_background_mask = tf.not_equal(tf.argmax(labels[0], -1), 0)

        # with tf.control_dependencies([
        #         tf.assert_none_equal(
        #             tf.reduce_sum(tf.to_float(non_background_mask)), 0)
        # ]):

        class_loss = focal_sigmoid_cross_entropy_with_logits(
            labels=labels[0], logits=logits[0])
        class_loss = tf.reduce_sum(class_loss) / tf.reduce_sum(
            tf.to_float(non_background_mask))

        regr_loss = tf.losses.huber_loss(
            labels[1], logits[1], reduction=tf.losses.Reduction.NONE)
        regr_loss = tf.reduce_sum(
            tf.boolean_mask(regr_loss, non_background_mask))

    return class_loss, regr_loss
