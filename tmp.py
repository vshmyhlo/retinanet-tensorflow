import argparse
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from level import make_levels
from utils import log_args


def make_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument('--ann-path', type=str, required=True)
  parser.add_argument('--dataset-path', type=str, required=True)
  parser.add_argument('--download', action='store_true')

  return parser


def main():
  args = make_parser().parse_args()
  log_args(args)

  num_classes = 20
  levels = make_levels()

  ds = make_dataset(
      ann_path=args.ann_path,
      dataset_path=args.dataset_path,
      num_classes=num_classes,
      levels=levels,
      download=args.download)
  iter = ds.make_one_shot_iterator()
  image, classifications_true, regressions_true = iter.get_next()

  image_with_boxmap = image
  # image_with_boxmap = tf.image.draw_bounding_boxes(
  #     tf.expand_dims(image, 0), tf.expand_dims(boxmap_true, 0))[0]

  with tf.Session() as sess:
    iwbm = sess.run(image_with_boxmap)
    plt.imshow(iwbm)
    plt.show()


if __name__ == '__main__':
  main()
