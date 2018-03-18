import augmentation
import tensorflow as tf
import numpy as np


class AugmentationTest(tf.test.TestCase):
    def test_flip(self):
        input = (
            tf.convert_to_tensor([
                [1, 2],
                [3, 4],
            ]),
            (tf.convert_to_tensor([
                [[1], [2]],
                [[3], [0]],
            ]), ),
            (tf.convert_to_tensor([
                [[[0., 0., .25, .25]], [[.25, .25, .5, .5]]],
                [[[0., 0., .25, .25]], [[0., 0., 0., 0.]]],
            ]), ),
        )

        actual = augmentation.flip(*input)

        expected = (
            tf.convert_to_tensor([
                [2, 1],
                [4, 3],
            ]),
            (tf.convert_to_tensor([
                [[2], [1]],
                [[0], [3]],
            ]), ),
            (tf.convert_to_tensor([
                [[[.5, .5, .75, .75]], [[.75, .75, 1., 1.]]],
                [[[0., 0., 0., 0.]], [[.75, .75, 1., 1.]]],
            ]), ),
        )

        a, e = self.evaluate([actual, expected])

        assert len(a) == len(e)
        for a, e in zip(a, e):
            assert np.array_equal(a, e)
