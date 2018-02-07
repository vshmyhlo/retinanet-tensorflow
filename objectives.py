# TODO; a-balancing
def focal_softmax_cross_entropy_with_logits(labels,
                                            logits,
                                            focus=2,
                                            dim=-1,
                                            name=None):
  loss = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, dim=dim, name=name)
  prob = tf.nn.softmax(logits)
  modulating_factor = (1 - prob)**focus

  return modulating_factor * loss
