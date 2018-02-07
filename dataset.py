import os
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO


def make_labels(img, levels, num_classes):
  image_size = np.array([img['height'], img['width']])

  classifications = [
      np.ones((*np.int32(np.ceil(image_size / 2**l.number)),
               num_classes * len(l.anchor_aspect_ratios)))
      for l in reversed(levels)
  ]
  regressions = [
      np.ones((*np.int32(np.ceil(image_size / 2**l.number)),
               4 * len(l.anchor_aspect_ratios))) for l in reversed(levels)
  ]

  return classifications, regressions


def gen(ann_path, dataset_path, levels, num_classes, download):
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
    image_path = os.path.join(dataset_path, img['file_name'])
    classifications, regressions = make_labels(
        img, levels=levels, num_classes=num_classes)

    yield image_path, tuple(classifications), tuple(regressions)

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
        num_classes=num_classes,
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
          tuple(tf.float32 for _ in levels),
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
