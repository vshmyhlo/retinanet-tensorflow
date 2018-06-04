import os
import numpy as np
import tensorflow as tf
import data_loaders.coco as coco
import utils
import augmentation
import argparse
import itertools
from tqdm import tqdm

NEG_IOU_THRESHOLD = 0.4
POS_IOU_THRESHOLD = 0.5
MEAN = [0.46618041, 0.44669811, 0.40252436]
STD = [0.27940595, 0.27489075, 0.28920765]


def position_grid(size):
    cell_size = tf.to_float(1 / size)

    y_pos = tf.linspace(cell_size[0] / 2, 1 - cell_size[0] / 2, size[0])
    x_pos = tf.linspace(cell_size[1] / 2, 1 - cell_size[1] / 2, size[1])

    x_pos, y_pos = tf.meshgrid(x_pos, y_pos)
    grid = tf.stack([y_pos, x_pos], -1)

    return grid


def to_center_box(box):
    a, b = tf.split(box, 2, -1)
    size = b - a

    return tf.concat([a + size / 2, size], -1)


def from_center_box(box):
    pos, size = tf.split(box, 2, -1)
    half_size = size / 2

    return tf.concat([pos - half_size, pos + half_size], -1)


def level_labels(image_size, class_id, true_box, level, factor, num_classes):
    num_objects = tf.shape(true_box)[0]
    num_anchors = level.anchor_sizes.shape[0]

    # [OBJECTS, 4]
    true_box = to_center_box(true_box)
    # [OBJECTS, 1, 1, 1, 4]
    true_box = tf.reshape(true_box, (num_objects, 1, 1, 1, 4))

    # [ANCHORS, 2]
    anchor_size = tf.to_float(level.anchor_sizes / image_size)

    grid_size = tf.to_int32(tf.ceil(image_size / factor))
    # [H, W, 2]
    anchor_position = position_grid(grid_size)
    h, w = grid_size[0], grid_size[1]
    del grid_size
    # [1, H, W, 1, 2]
    anchor_position = tf.reshape(anchor_position, (1, h, w, 1, 2))
    # [1, H, W, ANCHORS, 2]
    anchor_position = tf.tile(anchor_position, (1, 1, 1, num_anchors, 1))
    # [1, 1, 1, ANCHORS, 2]
    anchor_size = tf.reshape(anchor_size, (1, 1, 1, num_anchors, 2))
    # [1, H, W, ANCHORS, 2]
    anchor_size = tf.tile(anchor_size, (1, h, w, 1, 1))
    # [1, H, W, ANCHORS, 4]
    anchor = tf.concat([anchor_position, anchor_size], -1)

    # classification

    # [OBJECTS, H, W, ANCHORS]
    iou = utils.iou(from_center_box(anchor), from_center_box(true_box))
    # [H, W, ANCHORS]
    iou_index = tf.argmax(iou, 0)
    # [H, W, ANCHORS]
    iou_value = tf.reduce_max(iou, 0)

    # mask for assigning background class
    # [H, W, ANCHORS]
    bg_mask = iou_value < NEG_IOU_THRESHOLD
    # mask for ignoring unassigned anchors
    # [H, W, ANCHORS]
    not_ignored_mask = tf.logical_or(bg_mask, iou_value >= POS_IOU_THRESHOLD)

    # assign class labels to anchors
    # [H, W, ANCHORS]
    classification = tf.gather(class_id, iou_index)
    # [H, W, ANCHORS, CLASSES]
    classification = tf.one_hot(classification, num_classes)
    # assign background class to anchors with iou < NEG_IOU_THRESHOLD
    # [H, W, ANCHORS, CLASSES]
    bg_mask_expanded = tf.tile(tf.expand_dims(bg_mask, -1), (1, 1, 1, num_classes))
    # TODO: check if this is correct
    # [H, W, ANCHORS, CLASSES]
    classification = tf.where(bg_mask_expanded, tf.zeros_like(classification), classification)

    # regression

    # [OBJECTS, 1, 1, 1, 2], [OBJECTS, 1, 1, 1, 2],
    true_position, true_size = tf.split(true_box, 2, -1)

    # [OBJECTS, H, W, ANCHORS, 2]
    shifts = (true_position - anchor_position) / anchor_size
    # [OBJECTS, H, W, ANCHORS, 2]
    scales = true_size / anchor_size
    # [OBJECTS, H, W, ANCHORS, 4]
    regression = tf.concat([shifts, tf.log(scales)], -1)

    # select regression for assigned anchor
    # [H, W, ANCHORS, 1]
    iou_index_expanded = tf.expand_dims(iou_index, -1)
    # [OBJECTS, H, W, ANCHORS, 1]
    iou_index_expanded = tf.one_hot(iou_index_expanded, num_objects, axis=0)

    # [H, W, ANCHORS, 4]
    regression = tf.reduce_sum(regression * iou_index_expanded, 0)  # TODO: should mask bg?

    return classification, regression, not_ignored_mask


def build_labels(image_size, class_ids, boxes, levels, num_classes):
    labels = {
        pn: level_labels(
            image_size,
            class_ids,
            boxes,
            level=levels[pn],
            factor=2**int(pn[-1]),
            num_classes=num_classes)
        for pn in levels
    }

    classifications = {pn: labels[pn][0] for pn in labels}
    regressions = {pn: labels[pn][1] for pn in labels}
    not_ignored_masks = {pn: labels[pn][2] for pn in labels}

    return classifications, regressions, not_ignored_masks


def rescale_image(image, scale):
    size = tf.to_float(tf.shape(image)[:2])
    shorter = tf.argmin(size)
    ratio = scale / size[shorter]
    new_size = tf.to_int32(tf.round(size * ratio))

    return tf.image.resize_images(image, new_size, method=tf.image.ResizeMethod.BILINEAR, align_corners=True)


def build_dataset(ann_path, dataset_path, levels, download, augment, scale=None):
    def load_image_with_labels(input):
        image = tf.read_file(input['image_file'])
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image_size = tf.shape(image)[:2]
        boxes = input['boxes'] / tf.to_float(tf.concat([image_size, image_size], 0))

        if scale is not None:
            image = rescale_image(image, scale)
            image_size = tf.shape(image)[:2]

        classifications, regressions, not_ignored_masks = build_labels(
            image_size, input['class_ids'], boxes, levels=levels, num_classes=dl.num_classes)

        return {
            'image': image,
            'classifications': classifications,
            'regressions': regressions,
            'not_ignored_masks': not_ignored_masks
        }

    def preprocess(input):
        flipped = augmentation.flip(input)

        image = tf.stack([input['image'], flipped['image']], 0)
        classifications = {
            pn: tf.stack([input['classifications'][pn], flipped['classifications'][pn]], 0)
            for pn in input['classifications']}
        regressions = {
            pn: tf.stack([input['regressions'][pn], flipped['regressions'][pn]], 0)
            for pn in input['regressions']}
        not_ignored_masks = {
            pn: tf.stack([input['not_ignored_masks'][pn], flipped['not_ignored_masks'][pn]], 0)
            for pn in input['not_ignored_masks']}

        return {
            'image': image,
            'classifications': classifications,
            'regressions': regressions,
            'not_ignored_masks': not_ignored_masks
        }

    def augment_sample(input):
        # TODO: add augmentation
        # image = tf.image.random_contrast(image, 0.8, 1.2)
        # image = tf.image.random_brightness(image, 0.2)
        # image = tf.image.random_saturation(image, 0.8, 1.0)

        return input

    def mapper(input):
        input = load_image_with_labels(input)
        input = preprocess(input)

        if augment:
            input = augment_sample(input)

        return input

    dl = coco.COCO(ann_path, dataset_path, download)
    ds = tf.data.Dataset.from_generator(
        lambda: dl,
        output_types={'image_file': tf.string, 'class_ids': tf.int32, 'boxes': tf.float32},
        output_shapes={'image_file': [], 'class_ids': [None], 'boxes': [None, 4]})

    ds = ds.map(mapper, num_parallel_calls=min(os.cpu_count(), 8))

    return ds, dl.num_classes


def compute_mean_std():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs=2, required=True)

    args = parser.parse_args()
    ds, num_classes = build_dataset(
        ann_path=args.dataset[0],
        dataset_path=args.dataset[1],
        levels={},
        download=False,
        augment=False)
    iter = ds.make_initializable_iterator()
    image, classifications_true, regressions_true = iter.get_next()

    mean = np.array([0., 0., 0.])
    std = np.array([0., 0., 0.])
    i = 0

    with tf.Session() as sess:
        sess.run(iter.initializer)
        for _ in tqdm(itertools.count()):
            try:
                x = sess.run(image)
                i += x.shape[0] * x.shape[1] * x.shape[2]
                mean += x.sum((0, 1, 2))
            except tf.errors.OutOfRangeError:
                break

        mean = mean / i

        sess.run(iter.initializer)
        for _ in tqdm(itertools.count()):
            try:
                x = sess.run(image)
                std += ((x - mean)**2).sum((0, 1, 2))
            except tf.errors.OutOfRangeError:
                break

        std = np.sqrt(std / i)

    return mean, std


if __name__ == '__main__':
    mean, std = compute_mean_std()
    print('mean:', mean)
    print('std:', std)
