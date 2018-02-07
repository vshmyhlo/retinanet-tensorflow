import tensorflow as tf


# TODO: a-balancing
def focal_sigmoid_cross_entropy_with_logits(labels,
                                            logits,
                                            focus=2.0,
                                            dim=-1,
                                            name=None):
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels, logits=logits, name=name)
  prob = tf.nn.sigmoid(logits)
  modulating_factor = (1.0 - prob)**focus

  return modulating_factor * loss


def validate_output_shapes(true, pred):
  validations = []

  for ct, rt, cp, rp in zip(*true, *pred):
    validations.append(tf.assert_equal(tf.shape(ct), tf.shape(cp)))
    validations.append(tf.assert_equal(tf.shape(rt), tf.shape(rp)))

  return validations


def loss(true, pred):
  assert len(true[0]) == len(true[1]) == len(pred[0]) == len(pred[1])

  class_losses = []
  regr_losses = []

  with tf.control_dependencies(validate_output_shapes(true, pred)):
    for ct, rt, cp, rp in zip(*true, *pred):
      class_losses.append(
          tf.reduce_mean(
              focal_sigmoid_cross_entropy_with_logits(labels=ct, logits=cp)))
      regr_losses.append(tf.reduce_mean(rt - rp))

  class_loss = sum(class_losses)
  regr_loss = sum(regr_losses)

  return class_loss, regr_loss
