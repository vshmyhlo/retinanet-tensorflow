import augmentation
import tensorflow as tf
import numpy as np


class AugmentationTest(tf.test.TestCase):
    def test_flip(self):
        input = (
            [
                [1, 2],
                [3, 4]
            ], {'P3': [
                [[1], [2]],
                [[3], [4]],
            ]}, {'P3': [
                [[[0., 0., .5, .5]], [[.25, .25, .5, .5]]],
                [[[1., 1., .5, .5]], [[.75, .75, .25, .25]]],
            ]}, {'P3': [
                [[True], [False]],
                [[False], [True]]
            ]})

        actual = augmentation.flip(*input)

        expected = (
            [
                [2, 1],
                [4, 3],
            ], {'P3': [
                [[2], [1]],
                [[4], [3]],
            ]}, {'P3': [
                [[[.25, -.25, .5, .5]], [[0., 0., .5, .5]]],
                [[[.75, -.75, .25, .25]], [[1., -1., .5, .5]]],
            ]}, {'P3': [
                [[False], [True]],
                [[True], [False]]
            ]})

        actual = self.evaluate(actual)

        assert np.array_equal(actual[0], expected[0])
        assert np.array_equal(actual[1]['P3'], expected[1]['P3'])
        assert np.array_equal(actual[2]['P3'], expected[2]['P3'])
        assert np.array_equal(actual[3]['P3'], expected[3]['P3'])
