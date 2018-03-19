import tensorflow as tf
import utils
import numpy as np


class UtilsTest(tf.test.TestCase):
    def test_merge_outputs(self):
        outputs = [
            tf.ones((2, 3, 3, 4)),
            tf.ones((2, 5, 5, 4)),
            tf.ones((2, 7, 7, 4))
        ]

        merged = utils.merge_outputs(outputs)

        m = self.evaluate(merged)
        assert m.shape == (2, 83, 4)

    def test_boxmap_anchor_relative_to_image_relative(self):
        c = [[0.5, 1.0, 0.25, 0.75]]
        regression = tf.convert_to_tensor([
            [c, c, c, c],
            [c, c, c, c],
            [c, c, c, c],
        ])
        regression = tf.expand_dims(regression, 0)

        expected = [
            [
                [[1 / 6 + 0.5, 1 / 8 + 1.0, 0.25, 0.75]],
                [[1 / 6 + 0.5, 3 / 8 + 1.0, 0.25, 0.75]],
                [[1 / 6 + 0.5, 5 / 8 + 1.0, 0.25, 0.75]],
                [[1 / 6 + 0.5, 7 / 8 + 1.0, 0.25, 0.75]],
            ],
            [
                [[3 / 6 + 0.5, 1 / 8 + 1.0, 0.25, 0.75]],
                [[3 / 6 + 0.5, 3 / 8 + 1.0, 0.25, 0.75]],
                [[3 / 6 + 0.5, 5 / 8 + 1.0, 0.25, 0.75]],
                [[3 / 6 + 0.5, 7 / 8 + 1.0, 0.25, 0.75]],
            ],
            [
                [[5 / 6 + 0.5, 1 / 8 + 1.0, 0.25, 0.75]],
                [[5 / 6 + 0.5, 3 / 8 + 1.0, 0.25, 0.75]],
                [[5 / 6 + 0.5, 5 / 8 + 1.0, 0.25, 0.75]],
                [[5 / 6 + 0.5, 7 / 8 + 1.0, 0.25, 0.75]],
            ],
        ]
        expected = tf.expand_dims(expected, 0)

        actual = utils.boxmap_anchor_relative_to_image_relative(regression)

        a, e = self.evaluate([actual, expected])
        assert np.allclose(a, e)
        assert a.shape == (1, 3, 4, 1, 4)

    def test_anchor_boxmap(self):
        grid_size = tf.convert_to_tensor([3, 4])
        size1 = [0.2, 0.4]
        anchor_boxes = tf.convert_to_tensor([size1])

        expected = tf.convert_to_tensor([
            [
                [[1 / 6 - 0.1, 1 / 8 - 0.2, 1 / 6 + 0.1, 1 / 8 + 0.2]],
                [[1 / 6 - 0.1, 3 / 8 - 0.2, 1 / 6 + 0.1, 3 / 8 + 0.2]],
                [[1 / 6 - 0.1, 5 / 8 - 0.2, 1 / 6 + 0.1, 5 / 8 + 0.2]],
                [[1 / 6 - 0.1, 7 / 8 - 0.2, 1 / 6 + 0.1, 7 / 8 + 0.2]],
            ],
            [
                [[3 / 6 - 0.1, 1 / 8 - 0.2, 3 / 6 + 0.1, 1 / 8 + 0.2]],
                [[3 / 6 - 0.1, 3 / 8 - 0.2, 3 / 6 + 0.1, 3 / 8 + 0.2]],
                [[3 / 6 - 0.1, 5 / 8 - 0.2, 3 / 6 + 0.1, 5 / 8 + 0.2]],
                [[3 / 6 - 0.1, 7 / 8 - 0.2, 3 / 6 + 0.1, 7 / 8 + 0.2]],
            ],
            [
                [[5 / 6 - 0.1, 1 / 8 - 0.2, 5 / 6 + 0.1, 1 / 8 + 0.2]],
                [[5 / 6 - 0.1, 3 / 8 - 0.2, 5 / 6 + 0.1, 3 / 8 + 0.2]],
                [[5 / 6 - 0.1, 5 / 8 - 0.2, 5 / 6 + 0.1, 5 / 8 + 0.2]],
                [[5 / 6 - 0.1, 7 / 8 - 0.2, 5 / 6 + 0.1, 7 / 8 + 0.2]],
            ],
        ])
        expected = tf.expand_dims(expected, 0)
        actual = utils.anchor_boxmap(grid_size, anchor_boxes)

        a, e = self.evaluate([actual, expected])
        assert np.allclose(a, e)
        assert a.shape == (1, 3, 4, 1, 4)

    def test_boxmap_center_relative_to_corner_relative(self):
        c = [[0.5, 1.0, 0.2, 0.4]]
        regression = tf.convert_to_tensor([
            [c, c, c, c],
            [c, c, c, c],
            [c, c, c, c],
        ])
        regression = tf.expand_dims(regression, 0)

        c = [[0.4, 0.8, 0.6, 1.2]]
        expected = tf.convert_to_tensor([
            [c, c, c, c],
            [c, c, c, c],
            [c, c, c, c],
        ])
        expected = tf.expand_dims(expected, 0)

        actual = utils.boxmap_center_relative_to_corner_relative(regression)

        a, e = self.evaluate([actual, expected])
        assert np.array_equal(a, e)
        assert a.shape == (1, 3, 4, 1, 4)

    def test_iou(self):
        box_a = tf.convert_to_tensor([
            [0.1, 0.1, 0.2, 0.2],
            [100, 100, 200, 200],
            [0.1, 0.1, 0.2, 0.2],
            [1., 1., 1., 1.],
        ])
        box_b = tf.convert_to_tensor([
            [0.1, 0.1, 0.3, 0.3],
            [100, 100, 300, 300],
            [100, 100, 300, 300],
            [0., 0., 0., 0.],
        ])

        actual = utils.iou(box_a, box_b)
        expected = tf.convert_to_tensor([0.25, 0.25, 0, 0])

        a, e = self.evaluate([actual, expected])
        assert np.allclose(a, e)
        assert a.shape == (4, )
