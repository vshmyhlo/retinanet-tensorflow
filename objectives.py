import tensorflow as tf


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

    # a_balance = alpha * labels + (1 - alpha) * (1 - labels)
    a_balance = 1.0

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

    class_loss = focal_sigmoid_cross_entropy_with_logits(
        labels=labels[0], logits=logits[0])

    regr_loss = tf.losses.huber_loss(
        labels[1], logits[1], reduction=tf.losses.Reduction.NONE)
    regr_loss = tf.boolean_mask(regr_loss, non_background_mask)

  return class_loss, regr_loss, non_background_mask


# TODO: check why bounding box is not assigned to any anchor box
def global_mean(tensors, name='global_mean'):
  with tf.name_scope(name):
    global_sum = sum(tf.reduce_sum(t) for t in tensors)
    size = sum(tf.size(t) for t in tensors)
    size = tf.to_float(tf.maximum(size, 1))

    return global_sum / size


def global_sum(tensors, name='global_sum'):
  with tf.name_scope(name):
    return sum(tf.reduce_sum(tf.to_float(t)) for t in tensors)


def loss(true, pred, name='loss'):
  assert len(true[0]) == len(true[1]) == len(pred[0]) == len(pred[1])

  with tf.name_scope(name):
    non_background_masks = []
    class_losses = []
    regr_losses = []

    with tf.control_dependencies(validate_output_shapes(true, pred)):
      for ct, rt, cp, rp in zip(*true, *pred):
        class_loss, regr_loss, non_background_mask = level_loss((ct, rt),
                                                                (cp, rp))
        non_background_masks.append(non_background_mask)
        class_losses.append(class_loss)
        regr_losses.append(regr_loss)

    class_loss = global_sum(class_losses) / tf.maximum(
        global_sum(non_background_masks), 1)
    regr_loss = global_mean(regr_losses)

  return class_loss, regr_loss
