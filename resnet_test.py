import tensorflow as tf
import resnet


class ResNetTest(tf.test.TestCase):
    def test_resnet_v2_50(self):
        size = 224
        net = resnet.ResNeXt_50()
        input = tf.zeros((1, size, size, 3))
        output = net(input, False)

        for c in output:
            factor = int(c[-1])
            new_size = size // 2**factor

            if c == 'C1':
                width = 64
            else:
                width = int(64 * 2**factor)

            assert output[c].shape == (1, new_size, new_size, width)
