import tensorflow as tf


def flip(image, classifications, regressions, ignored_mask):
    image = tf.reverse(image, [1])
    classifications = {pn: tf.reverse(classifications[pn], [1]) for pn in classifications}
    regressions = {pn: tf.reverse(regressions[pn], [1]) for pn in regressions}
    ignored_mask = {pn: tf.reverse(ignored_mask[pn], [1]) for pn in ignored_mask}
    for pn in regressions:
        y, x, h, w = tf.split(regressions[pn], 4, -1)
        regressions[pn] = tf.concat([y, -x, h, w], -1)

    return image, classifications, regressions, ignored_mask
