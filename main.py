import argparse
import tensorflow as tf
import retinanet
from utils import log_args
from level import make_levels
import objectives
import dataset

# TODO: prior class prob initialization
# TODO: resize to 800
# TODO: sigmoid and exp for regression


def make_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=32)
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

  # print()
  # print(*classifications_true, sep='\n')
  # print()
  # print(*regressions_true, sep='\n')

  ds, num_classes = dataset.make_dataset(
      ann_path=args.ann_path,
      dataset_path=args.dataset_path,
      levels=levels,
      download=False)

  ds = ds.batch(1)
  assert num_classes == 80

  iter = ds.make_one_shot_iterator()
  image, classifications_true, regressions_true = iter.get_next()

  classifications_pred, regressions_pred = retinanet.retinaneet(
      image,
      num_classes=num_classes,
      levels=levels,
      dropout=args.dropout,
      weight_decay=args.weight_decay,
      training=training)

  loss = objectives.loss((classifications_true, regressions_true),
                         (classifications_pred, regressions_pred))

  # print()
  # print(*classifications_pred, sep='\n')
  # print()
  # print(*regressions_pred, sep='\n')

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tmp = sess.run([classifications_pred, regressions_pred], {training: False})
    print([[x.shape for x in xs] for xs in tmp])

    tmp = sess.run(loss, {training: False})
    print(tmp)


if __name__ == '__main__':
  main()
