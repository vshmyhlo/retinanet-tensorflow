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
