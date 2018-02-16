import tensorflow as tf


# TODO: a-balancing
def focal_sigmoid_cross_entropy_with_logits(
    labels,
    logits,
    focus=2.0,
    alpha=0.25,
    dim=-1,
    name='focal_sigmoid_cross_entropy_with_logits'):
  with tf.name_scope(name):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)

    a_balance = alpha * labels + (1 - alpha) * (1 - labels)

    prob = tf.nn.sigmoid(logits)
    prob_true = prob * labels + (1 - prob) * (1 - labels)
    modulating_factor = (1.0 - prob_true)**focus

    return a_balance * modulating_factor * loss


def validate_output_shapes(true, pred, name='validate_output_shapes'):
  validations = []

  with tf.name_scope(name):
    for ct, rt, cp, rp in zip(*true, *pred):
      validations.append(tf.assert_equal(tf.shape(ct), tf.shape(cp)))
      validations.append(tf.assert_equal(tf.shape(rt), tf.shape(rp)))

  return validations


def level_loss(labels, logits, name='level_loss'):
  with tf.name_scope(name):
    non_background_mask = tf.not_equal(tf.argmax(labels[0], -1), 0)
    # non_background_mask = tf.expand_dims(non_background_mask, -1)
    # non_background_mask = tf.to_float(non_background_mask)

    class_loss = focal_sigmoid_cross_entropy_with_logits(
        labels=labels[0], logits=logits[0])
    # class_loss = tf.reduce_mean(class_loss)

    regr_loss = tf.square(labels[1] - logits[1])
    regr_loss = tf.boolean_mask(regr_loss, non_background_mask)

    # regr_loss = tf.cond(
    #     tf.reduce_sum(tf.to_float(non_background_mask)) > 0,
    #     lambda: tf.reduce_mean(tf.boolean_mask(regr_loss, non_background_mask)),
    #     lambda: 0.0)

  return class_loss, regr_loss


def global_mean(tensors, name='global_mean'):
  with tf.name_scope(name):
    size = sum(tf.size(t) for t in tensors)
    global_sum = sum(tf.reduce_sum(t) for t in tensors)

    # TODO:
    return tf.cond(size > 0, lambda: global_sum / tf.to_float(size),
                   lambda: 0.0)

    # with tf.control_dependencies([tf.assert_positive(size)]):
    #   return global_sum / tf.to_float(size)


def loss(true, pred, name='loss'):
  assert len(true[0]) == len(true[1]) == len(pred[0]) == len(pred[1])

  with tf.name_scope(name):
    class_losses = []
    regr_losses = []

    with tf.control_dependencies(validate_output_shapes(true, pred)):
      for ct, rt, cp, rp in zip(*true, *pred):
        class_loss, regr_loss = level_loss((ct, rt), (cp, rp))
        class_losses.append(class_loss)
        regr_losses.append(regr_loss)

    class_loss = global_mean(class_losses)
    regr_loss = global_mean(regr_losses)

  return class_loss, regr_loss
