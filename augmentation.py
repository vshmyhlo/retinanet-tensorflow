import tensorflow as tf


# TODO: add tests
def flip(image, classifications, regressions):
    image = tf.reverse(image, [1])
    classifications = {
        pn: tf.reverse(classifications[pn], [1])
        for pn in classifications
    }
    regressions = {pn: tf.reverse(regressions[pn], [1]) for pn in regressions}

    return image, classifications, regressions
