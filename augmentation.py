import tensorflow as tf


# TODO: add tests
def flip(image, classifications, regressions):
    image = tf.reverse(image, [1])
    classifications = {
        pn: tf.reverse(classifications[pn], [1])
        for pn in classifications
    }
    regressions = {pn: tf.reverse(regressions[pn], [1]) for pn in regressions}
    for pn in regressions:
        print(regressions[pn])
        y1, x1, y2, x2 = tf.split(regressions[pn], 4, -1)
        regressions[pn] = tf.concat([y1, 1 - x2, y2, 1 - x1], -1)

    return image, classifications, regressions
