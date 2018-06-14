import tensorflow as tf
import numpy as np
import losses


class ObjectivesTest(tf.test.TestCase):
    def test_merge_outputs(self):
        outputs = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        trainable_mask = {'a': [False, True, True], 'b': [True, True, False]}
        merged = losses.merge_outputs(outputs, trainable_mask)

        actual = self.evaluate(merged)
        expected = [2, 3, 4, 5]

        assert np.array_equal(actual, expected)

    def test_regression_loss(self):
        logits = [[1], [2], [3]]
        labels = [[3], [4], [6]]
        non_background_mask = [True, False, True]

        loss = losses.regression_loss(labels=labels, logits=logits, non_bg_mask=non_background_mask)

        actual = self.evaluate(loss)
        expected = 2.0

        assert np.array_equal(actual, expected)
