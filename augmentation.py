import tensorflow as tf


def flip(image, classifications, regressions):
    image = tf.reverse(image, [1])
    classifications = tuple(tf.reverse(x, [1]) for x in classifications)
    regressions = tuple(1 - tf.reverse(x, [1]) for x in regressions)

    # TODO: refactor this mess
    regressions = tuple(
        tf.where(
            tf.tile(tf.expand_dims(tf.not_equal(c, 0), -1), (1, 1, 1, 4)),
            tf.concat([r[..., 2:], r[..., :2]], -1),
            tf.zeros_like(r),
        ) for c, r in zip(classifications, regressions))

    return image, classifications, regressions
