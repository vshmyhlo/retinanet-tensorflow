import tensorflow as tf
import retinanet
import numpy as np


class RetinanetTest(tf.test.TestCase):
    def test_validate_lateral_shape(self):
        input = tf.ones((1, 15, 15, 3))
        lateral = tf.ones((1, 30, 30, 3))

        assertion = retinanet.validate_lateral_shape(input, lateral)

        self.evaluate(assertion)

    def test_scale_regression(self):
        regression = tf.convert_to_tensor([
            [0.5, 1.0, 0.5, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ])
        regression = tf.reshape(regression, (1, 1, 1, 2, 4))

        anchor_boxes = tf.convert_to_tensor([
            [0.2, 0.4],
            [0.4, 0.2],
        ])

        actual = retinanet.scale_regression(regression, anchor_boxes)

        expected = tf.convert_to_tensor([
            [0.1, 0.4, 0.1, 0.4],
            [0.2, 0.1, 0.2, 0.1],
        ])
        expected = tf.reshape(expected, (1, 1, 1, 2, 4))

        a, e = self.evaluate([actual, expected])
        assert np.array_equal(a, e)
        assert a.shape == (1, 1, 1, 2, 4)

    def test_regression_postprocess(self):
        anchor_boxes = tf.convert_to_tensor([[.5, .5]])
        regression = tf.convert_to_tensor([
            [[[.5, .5, 1., 1.]], [[0., 0., 0., 0.]]],
            [[[0., 0., 0., 0.]], [[-.5, -.5, 2., 2.]]],
        ])
        regression = tf.expand_dims(regression, 0)

        expected = tf.convert_to_tensor([
            [[[.25, .25, .75, .75]], [[.25, .75, .25, .75]]],
            [[[.75, .25, .75, .25]], [[0., 0., 1., 1.]]],
        ])
        expected = tf.expand_dims(expected, 0)

        actual = retinanet.regression_postprocess(regression, anchor_boxes)

        a, e = self.evaluate([actual, expected])
        assert np.array_equal(a, e)
        assert a.shape == (1, 2, 2, 1, 4)
