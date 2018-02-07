import os
import argparse
import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO
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


def gen(ann_path, dataset_path, levels, download):
  coco = COCO(ann_path)
  if download:
    coco.download(tarDir=dataset_path)

  # for img_
  # ann_ids = coco.getAnnIds()
  # print(len(ann_ids))
  # fail

  # print(len(cats))

  # cat_ids = coco.getCatIds()
  # assert len(cat_ids) == 80
  # coco.loadCats(cat_ids)
  # img_ids = coco.getImgIds(catIds=cat_ids)

  img_ids = coco.getImgIds()

  # if img_id = in
  imgs = coco.loadImgs(ids=img_ids)

  for img in imgs:
    yield (
        os.path.join(dataset_path, img['file_name']),
        tuple(np.ones((32, 32, 60)) for _ in levels),
        tuple(np.ones((32, 32, 12)) for _ in levels),
    )

  # print(len(imgs))
  # fail

  # assert len(imgs) == 118287
  # print(imgs[0])
  # # print(imgs[0])
  # anns = coco.loadAnns()
  # print(len(anns))
  # print(anns[:5])
  #
  # ann_ids = coco.getAnnIds()
  # print(len(ann_ids))
  # anns = coco.loadAnns(ann_ids)
  # print(len(anns))
  # print(anns[0].keys())

  # print(len(img_ids))

  # print(len(img_ids))

  # img = imgs[0]
  # ann_ids = coco.getAnnIds(imgIds=img['id'])
  # anns = coco.loadAnns(ann_ids)
  #
  # image = Image.open(os.path.join(args.dataset_path, img['file_name']))
  # image = (np.array(image) / 255).astype(np.float32)

  # print(len(cat_ids))
  # print()
  # print(cat_ids)
  # print(cats)

  # nms = [cat['name'] for cat in cats]
  # print('COCO categories: \n{}\n'.format(' '.join(nms)))
  #
  # nms = set([cat['supercategory'] for cat in cats])
  # print('COCO supercategories: \n{}'.format(' '.join(nms)))

  assert image.shape[0] == img['height'], image.shape[1] == img['width']

  print(anns[0])
  boxes = [[
      ann['bbox'][0] / img['width'], ann['bbox'][1] / img['height'],
      ann['bbox'][2] / img['width'], ann['bbox'][3] / img['height']
  ] for ann in anns]

  boxmap_true = np.array(boxes, dtype=np.float32)

  boxmap_true = tf.stack([
      boxmap_true[:, 1],
      boxmap_true[:, 0],
      boxmap_true[:, 3] + boxmap_true[:, 1],
      boxmap_true[:, 2] + boxmap_true[:, 0],
  ], -1)


def make_dataset(ann_path, dataset_path, num_classes, levels, download):
  def make_gen():
    return gen(
        ann_path=ann_path,
        dataset_path=dataset_path,
        levels=levels,
        download=download)

  def load_image(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

  ds = tf.data.Dataset.from_generator(
      make_gen,
      output_types=(
          tf.string,
          tuple(tf.int32 for _ in levels),
          tuple(tf.float32 for _ in levels),
      ),
      output_shapes=(
          [],
          tuple([None, None,
                 len(l.anchor_aspect_ratios) * num_classes] for l in levels),
          tuple([None, None, len(l.anchor_aspect_ratios) * 4] for l in levels),
      ))

  ds = ds.map(lambda filename, *rest: (load_image(filename), *rest))

  return ds


def main():
  args = make_parser().parse_args()
  log_args(args)

  # fpn_levels = (3, 4, 5, 6, 7)
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
