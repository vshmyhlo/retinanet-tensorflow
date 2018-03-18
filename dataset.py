import os
import numpy as np
import tensorflow as tf
from coco import COCO
import utils

IOU_THRESHOLD = 0.5


# TODO: resize image
# TODO: background category
# TODO: ignored boxes
# TODO: regression exp
def level_labels(image_size, class_ids, boxes, level):
    grid_size = tf.to_int32(tf.ceil(image_size / 2**level.number))
    anchor_boxes = tf.to_float(level.anchor_boxes / image_size)

    # extract targets ##########################################################

    # [OBJECTS]
    classes_true = tf.concat([[0], class_ids], 0)
    # [OBJECTS, 4]
    boxes_true = tf.concat([[[0, 0, 0, 0]], boxes], 0)
    boxes_true = tf.to_float(
        boxes_true / tf.concat([image_size, image_size], 0))

    # compute iou ##############################################################

    # [OBJECTS, 1, 1, 1, 4]
    boxes_true_shape = tf.shape(boxes_true)
    boxes_true = tf.reshape(boxes_true, (boxes_true_shape[0], 1, 1, 1, 4))

    # [1, H, W, SIZES, 4]
    anchor_boxmap = utils.anchor_boxmap(grid_size, anchor_boxes)

    # [OBJECTS, H, W, SIZES]
    iou = utils.iou(anchor_boxmap, boxes_true)
    iou = tf.where(iou > IOU_THRESHOLD, iou, tf.zeros_like(iou))
    # for the given anchor box, finds the ground truth box with the highest iou
    # [H, W, SIZES]
    indices = tf.argmax(iou, 0)
    del iou

    # build classification targets #############################################

    # [H, W, SIZES]
    classification = tf.gather(classes_true, indices)

    # build regression targets #################################################

    # [H, W, SIZES, 1]
    indices_expanded = tf.expand_dims(indices, -1)
    # [OBJECTS, H, W, SIZES, 1]
    indices_expanded = tf.one_hot(
        indices_expanded, boxes_true_shape[0], axis=0)
    del indices

    # [OBJECTS, H, W, SIZES, 4]
    regression = boxes_true * indices_expanded
    # [H, W, SIZES, 4]
    regression = tf.reduce_sum(regression, 0)

    return classification, regression


def make_labels(image_size, class_ids, boxes, levels):
    labels = [
        level_labels(image_size, class_ids, boxes, level=level)
        for level in levels
    ]

    classifications, regressions = tuple(zip(*labels))

    return classifications, regressions


def gen(coco):
    for img in coco.load_imgs(coco.get_img_ids()):
        filename = os.path.join(coco.dataset_path,
                                img.filename).encode('utf-8')
        anns = coco.load_anns(coco.get_ann_ids(img_ids=img.id))
        class_ids = np.array([item.category_id for item in anns])
        boxes = np.array([item.box for item in anns])

        assert class_ids.shape[0] != 0
        assert boxes.shape[0] != 0

        yield filename, class_ids, boxes


def make_dataset(ann_path, dataset_path, levels, scale, shuffle, download):
    def one_hot(classifications):
        return tuple(tf.one_hot(x, coco.num_classes) for x in classifications)

    def load_image_with_labels(filename, class_ids, boxes):
        def load_image(filename):
            image = tf.read_file(filename)
            image = tf.image.decode_png(image, channels=3)
            image = tf.to_float(image) - 255 / 2

            return image

        image = load_image(filename)
        image_size = tf.shape(image)[:2]
        classifications, regressions = make_labels(
            image_size, class_ids, boxes, levels=levels)

        return image, classifications, regressions

    def preprocess(image, classifications, regressions):
        def flip(image, classifications, regressions):
            # TODO: add flipping

            # image = tf.reverse(image, [1])
            # classifications = tuple(
            #     tf.reverse(x, [1]) for x in classifications)
            # regressions = tuple(tf.reverse(x, [1]) for x in regressions)
            # regressions = tuple(
            #     tf.concat([x[..., :1], -x[..., 1:2], x[..., 2:]], -1)
            #     for x in regressions)

            return image, classifications, regressions

        classifications = one_hot(classifications)
        image_flipped, classifications_flipped, regressions_flipped = flip(
            image, classifications, regressions)

        image = tf.stack([image, image_flipped], 0)
        classifications = tuple(
            tf.stack([x, x_flipped], 0)
            for x, x_flipped in zip(classifications, classifications_flipped))
        regressions = tuple(
            tf.stack([x, x_flipped], 0)
            for x, x_flipped in zip(regressions, regressions_flipped))

        return image, classifications, regressions

    coco = COCO(ann_path, dataset_path, download)
    ds = tf.data.Dataset.from_generator(
        lambda: gen(coco),
        output_types=(tf.string, tf.int32, tf.int32),
        output_shapes=([], [None], [None, 4]))

    if shuffle is not None:
        ds = ds.shuffle(shuffle)

    ds = ds.map(load_image_with_labels)
    ds = ds.map(preprocess)

    return ds, coco.num_classes
