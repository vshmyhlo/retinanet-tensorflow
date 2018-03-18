import dataset
from level import Level
import numpy as np
import tensorflow as tf


class DatasetTest(tf.test.TestCase):
    def test_level_labels(self):
        image_size = tf.convert_to_tensor([32, 32], dtype=tf.int32)
        class_ids = tf.convert_to_tensor([100, 200, 300, 400])
        boxes = tf.convert_to_tensor([
            [0, 0, 16, 16],
            [8, 8, 24, 24],
            [16, 16, 32, 32],
            [-4, -4, 20, 20],
        ])
        level = Level(4, 16, [(1, 1)], [1, 1.5])

        actual = dataset.level_labels(image_size, class_ids, boxes, level)
        expected = (
            tf.convert_to_tensor([
                [[100, 400], [0, 0]],
                [[0, 0], [300, 0]],
            ]),
            tf.convert_to_tensor([
                [
                    [[0., 0., .5, .5], [-.125, -.125, .625, .625]],
                    [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                ],
                [
                    [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                    [[.5, .5, 1., 1.], [0., 0., 0., 0.]],
                ],
            ]),
        )

        a, e = self.evaluate([actual, expected])
        assert np.array_equal(a[0], e[0])
        assert a[0].shape == (2, 2, 2)
        assert np.array_equal(a[1], e[1])
        assert a[1].shape == (2, 2, 2, 4)
