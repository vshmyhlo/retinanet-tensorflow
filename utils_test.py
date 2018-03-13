import tensorflow as tf
import numpy as np
import utils


class UtilsTest(tf.test.TestCase):
    def test_merge_outputs(self):
        outputs = [
            tf.ones((2, 3, 3, 4)),
            tf.ones((2, 5, 5, 4)),
            tf.ones((2, 7, 7, 4))
        ]

        merged = utils.merge_outputs(outputs)

        with self.test_session():
            assert merged.eval().shape == (2, 83, 4)

    def test_boxmap_anchor_relative_to_image_relative(self):
        b = [[0.1, 0.2, 0.8, 0.9], [0.3, 0.4, 0.9, 0.8]]

        boxmap = tf.constant([
            [b, b, b, b],
            [b, b, b, b],
            [b, b, b, b],
        ])

        expected = [
            [
                [[1 / 6 + 0.1, 1 / 8 + 0.2, 0.8, 0.9],
                 [1 / 6 + 0.3, 1 / 8 + 0.4, 0.9, 0.8]],
                [[1 / 6 + 0.1, 3 / 8 + 0.2, 0.8, 0.9],
                 [1 / 6 + 0.3, 3 / 8 + 0.4, 0.9, 0.8]],
                [[1 / 6 + 0.1, 5 / 8 + 0.2, 0.8, 0.9],
                 [1 / 6 + 0.3, 5 / 8 + 0.4, 0.9, 0.8]],
                [[1 / 6 + 0.1, 7 / 8 + 0.2, 0.8, 0.9],
                 [1 / 6 + 0.3, 7 / 8 + 0.4, 0.9, 0.8]],
            ],
            [
                [[3 / 6 + 0.1, 1 / 8 + 0.2, 0.8, 0.9],
                 [3 / 6 + 0.3, 1 / 8 + 0.4, 0.9, 0.8]],
                [[3 / 6 + 0.1, 3 / 8 + 0.2, 0.8, 0.9],
                 [3 / 6 + 0.3, 3 / 8 + 0.4, 0.9, 0.8]],
                [[3 / 6 + 0.1, 5 / 8 + 0.2, 0.8, 0.9],
                 [3 / 6 + 0.3, 5 / 8 + 0.4, 0.9, 0.8]],
                [[3 / 6 + 0.1, 7 / 8 + 0.2, 0.8, 0.9],
                 [3 / 6 + 0.3, 7 / 8 + 0.4, 0.9, 0.8]],
            ],
            [
                [[5 / 6 + 0.1, 1 / 8 + 0.2, 0.8, 0.9],
                 [5 / 6 + 0.3, 1 / 8 + 0.4, 0.9, 0.8]],
                [[5 / 6 + 0.1, 3 / 8 + 0.2, 0.8, 0.9],
                 [5 / 6 + 0.3, 3 / 8 + 0.4, 0.9, 0.8]],
                [[5 / 6 + 0.1, 5 / 8 + 0.2, 0.8, 0.9],
                 [5 / 6 + 0.3, 5 / 8 + 0.4, 0.9, 0.8]],
                [[5 / 6 + 0.1, 7 / 8 + 0.2, 0.8, 0.9],
                 [5 / 6 + 0.3, 7 / 8 + 0.4, 0.9, 0.8]],
            ],
        ]

        actual = utils.boxmap_anchor_relative_to_image_relative(boxmap)

        with self.test_session():
            assert np.allclose(actual.eval(), expected)
