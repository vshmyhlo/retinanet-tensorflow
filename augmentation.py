import tensorflow as tf
import utils


def flip(input):
    image = tf.reverse(input['image'], [1])
    classifications = utils.dict_map(lambda x: tf.reverse(x, [1]), input['detection']['classifications'])
    regressions = utils.dict_map(lambda x: tf.reverse(x, [1]), input['detection']['regressions'])
    trainable_masks = utils.dict_map(lambda x: tf.reverse(x, [1]), input['trainable_masks'])

    for pn in regressions:
        y, x, h, w = tf.unstack(regressions[pn], axis=-1)
        regressions[pn] = tf.stack([y, -x, h, w], -1)

    return {
        'image': image,
        'detection': {
            'classifications': classifications,
            'regressions': regressions,
        },
        'trainable_masks': trainable_masks
    }
