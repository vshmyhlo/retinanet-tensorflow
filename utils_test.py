import tensorflow as tf
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
