import tensorflow as tf


def flip(input):
    image = tf.reverse(input['image'], [1])
    classifications = {pn: tf.reverse(input['classifications'][pn], [1]) for pn in input['classifications']}
    regressions = {pn: tf.reverse(input['regressions'][pn], [1]) for pn in input['regressions']}
    not_ignored_masks = {pn: tf.reverse(input['not_ignored_masks'][pn], [1]) for pn in input['not_ignored_masks']}
    for pn in regressions:
        y, x, h, w = tf.split(regressions[pn], 4, -1)
        regressions[pn] = tf.concat([y, -x, h, w], -1)

    return {
        'image': image,
        'classifications': classifications,
        'regressions': regressions,
        'not_ignored_masks': not_ignored_masks
    }
