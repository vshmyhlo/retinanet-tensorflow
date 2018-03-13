import termcolor
import tensorflow as tf


def success(str):
    return termcolor.colored(str, 'green')


def warning(str):
    return termcolor.colored(str, 'yellow')


def danger(str):
    return termcolor.colored(str, 'red')


def log_args(args):
    print(warning('arguments:'))
    for key, value in sorted(vars(args).items(), key=lambda kv: kv[0]):
        print(warning('\t{}:').format(key), value)


def merge_outputs(tensors, name='merge_outputs'):
    with tf.name_scope(name):
        validate_shapes = [
            tf.assert_greater_equal(tf.rank(t), 4) for t in tensors
        ]
        with tf.control_dependencies(validate_shapes):
            reshaped = []
            for t in tensors:
                sh = tf.shape(t)
                sh = tf.concat([[sh[0], sh[1] * sh[2]], sh[3:]], 0)
                reshaped.append(tf.reshape(t, sh))

            return tf.concat(reshaped, 1)


def boxmap_anchor_relative_to_image_relative(boxmap):
    grid_size = tf.shape(boxmap)[:2]
    cell_size = tf.to_float(1 / grid_size)

    y_pos = tf.linspace(cell_size[0] / 2, 1 - cell_size[0] / 2, grid_size[0])
    x_pos = tf.linspace(cell_size[1] / 2, 1 - cell_size[1] / 2, grid_size[1])

    x_pos, y_pos = tf.meshgrid(x_pos, y_pos)
    pos = tf.stack([y_pos, x_pos], -1)
    pos = tf.expand_dims(pos, -2)

    return tf.concat([boxmap[..., :2] + pos, boxmap[..., 2:]], -1)
