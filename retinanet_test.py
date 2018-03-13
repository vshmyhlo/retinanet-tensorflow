import tensorflow as tf
import retinanet


class RetinanetTest(tf.test.TestCase):
    def test_validate_lateral_shape(self):
        input = tf.ones((1, 15, 15, 3))
        lateral = tf.ones((1, 30, 30, 3))

        assertion = retinanet.validate_lateral_shape(input, lateral)

        with self.test_session() as sess:
            sess.run(assertion)
