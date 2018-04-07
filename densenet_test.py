import tensorflow as tf
import densenet


class DenseNetTest(tf.test.TestCase):
    def test_densenet_bc_169_output_shape(self):
        size = 224
        net = densenet.DenseNetBC_169(dropout_rate=0.2)
        input = tf.zeros((1, size, size, 3))
        output = net(input, False)

        for c in output:
            factor = int(c[-1])
            new_size = size // 2**factor
            assert output[c].shape[0:3] == (1, new_size, new_size)
