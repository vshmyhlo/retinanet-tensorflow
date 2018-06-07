import tensorflow as tf


def flip(input):
    image = tf.reverse(input['image'], [1])
    classifications = {pn: tf.reverse(input['classifications'][pn], [1]) for pn in input['classifications']}
    regressions = {pn: tf.reverse(input['regressions'][pn], [1]) for pn in input['regressions']}
    trainable_masks = {pn: tf.reverse(input['trainable_masks'][pn], [1]) for pn in input['trainable_masks']}
    for pn in regressions:
        y, x, h, w = tf.split(regressions[pn], 4, -1)
        regressions[pn] = tf.concat([y, -x, h, w], -1)

    return {
        'image': image,
        'classifications': classifications,
        'regressions': regressions,
        'trainable_masks': trainable_masks
    }
