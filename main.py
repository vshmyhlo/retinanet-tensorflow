import argparse
import itertools
import tensorflow as tf
import retinanet
from utils import log_args
from level import make_levels
import objectives
import dataset
import numpy as np

# TODO: prior class prob initialization
# TODO: resize to 800
# TODO: sigmoid and exp for regression
# TODO: rounding and float32


def draw_bounding_boxes(image, regressions, classifications, levels):
  image_size = tf.shape(image)[:2]
  image = tf.expand_dims(image, 0)

  for regression, classification, level in zip(regressions, classifications,
                                               levels):
    mask = tf.not_equal(tf.argmax(classification, -1), 0)
    anchor_boxes = tf.to_float(
        tf.stack([
            np.array(dataset.box_size(level.anchor_size, ratio)) / image_size
            for ratio in level.anchor_aspect_ratios
        ], 0))

    anchor_boxes = tf.reshape(anchor_boxes, (1, 1, -1, 2))

    grid_size = tf.shape(regression)[:2]
    cell_size = tf.to_float(1 / grid_size)
    y_pos = tf.reshape(
        tf.linspace(cell_size[0] / 2, 1 - cell_size[0] / 2, grid_size[0]),
        (-1, 1, 1, 1))
    x_pos = tf.reshape(
        tf.linspace(cell_size[1] / 2, 1 - cell_size[1] / 2, grid_size[1]),
        (1, -1, 1, 1))

    boxes = tf.concat([
        regression[..., 0:1] * anchor_boxes[..., 0:1] + y_pos,
        regression[..., 1:2] * anchor_boxes[..., 1:2] + x_pos,
        regression[..., 2:] * anchor_boxes,
    ], -1)
    boxes = tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2
    ], -1)
    boxes = tf.boolean_mask(boxes, mask)
    boxes = tf.expand_dims(boxes, 0)

    image = tf.image.draw_bounding_boxes(image, boxes)

  image = tf.squeeze(image, 0)
  return image


def make_parser():
  parser = argparse.ArgumentParser()
  # parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--learning-rate', type=float, default=1e-3)
  parser.add_argument('--weight-decay', type=float, default=1e-4)
  parser.add_argument('--dropout', type=float, default=0.2)
  parser.add_argument('--ann-path', type=str, required=True)
  parser.add_argument('--dataset-path', type=str, required=True)

  return parser


def main():
  args = make_parser().parse_args()
  log_args(args)

  levels = make_levels()
  training = tf.placeholder(tf.bool, [], name='training')
  global_step = tf.get_variable('global_step', initializer=0, trainable=False)

  ds, num_classes = dataset.make_dataset(
      ann_path=args.ann_path,
      dataset_path=args.dataset_path,
      levels=levels,
      download=False)

  ds = ds.batch(1)
  assert num_classes == 80 + 1

  iter = ds.make_one_shot_iterator()
  image, classifications_true, regressions_true = iter.get_next()

  classifications_pred, regressions_pred = retinanet.retinaneet(
      image,
      num_classes=num_classes,
      levels=levels,
      dropout=args.dropout,
      weight_decay=args.weight_decay,
      training=training)

  class_loss, regr_loss = objectives.loss(
      (classifications_true, regressions_true),
      (classifications_pred, regressions_pred))

  loss = class_loss + regr_loss
  train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(
      loss, global_step=global_step)

  image_with_boxes = draw_bounding_boxes(
      image[0], [y[0] for y in regressions_true],
      [y[0] for y in classifications_true], levels)

  with tf.name_scope('summary'):
    merged = tf.summary.merge(
        [tf.summary.image('boxmap', tf.expand_dims(image_with_boxes, 0))])

  saver = tf.train.Saver()

  with tf.Session() as sess, tf.summary.FileWriter(
      logdir='./tf_log/train', graph=sess.graph) as train_writer:
    restore_path = tf.train.latest_checkpoint(args.experiment_path)
    if restore_path:
      saver.restore(sess, restore_path)
    else:
      sess.run(tf.global_variables_initializer())

    for _ in itertools.count():
      _, step, cl, rl, summ = sess.run(
          [train_step, global_step, class_loss, regr_loss, merged], {
              training: True
          })

      print('step: {}, class_loss: {}, regr_loss: {}'.format(step, cl, rl))
      train_writer.add_summary(summ, step)
      saver.save(sess, './tf_log/train/model.ckpt')


if __name__ == '__main__':
  main()
