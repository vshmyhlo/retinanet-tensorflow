import os
import numpy as np
import tensorflow as tf
from coco import COCO
import cv_utils
from itertools import product

IOU_THRESHOLD = 0.5


def compute_box_size(base_size, aspect_ratio, scale_ratio):
    h = np.sqrt(base_size**2 / (
        aspect_ratio[0] * aspect_ratio[1])) * aspect_ratio[0] * scale_ratio
    w = np.sqrt(base_size**2 / (
        aspect_ratio[0] * aspect_ratio[1])) * aspect_ratio[1] * scale_ratio

    return h, w


# TODO: resize image
# TODO: background category
# TODO: ignored boxes
# TODO: regression exp
def make_level_labels(image, anns, level, num_classes):
    grid_size = np.int32(np.ceil(image.size / 2**level.number))

    # build grid anchor positions ##############################################
    cell_size = image.size / grid_size
    grid_y_positions = np.linspace(
        cell_size[0] / 2, image.size[0] - cell_size[0] / 2,
        grid_size[0]).reshape((grid_size[0], 1))  # [H, 1]
    grid_x_positions = np.linspace(
        cell_size[1] / 2, image.size[1] - cell_size[1] / 2,
        grid_size[1]).reshape((1, grid_size[1]))  # [1, W]
    del cell_size

    grid_y_positions = np.tile(grid_y_positions, (1, grid_size[1]))  # [H, W]
    grid_x_positions = np.tile(grid_x_positions, (grid_size[0], 1))  # [H, W]
    assert grid_x_positions.shape == grid_y_positions.shape

    grid_anchor_positions = np.stack([grid_y_positions, grid_x_positions],
                                     -1)  # [H, W, 2]
    del grid_x_positions, grid_y_positions

    # build grid anchor sizes ##################################################
    grid_anchor_sizes = np.array(
        [
            compute_box_size(level.anchor_size, aspect_ratio, scale_ratio)
            for aspect_ratio, scale_ratio in product(
                level.anchor_aspect_ratios, level.anchor_scale_ratios)
        ],
        dtype=np.float32)  # [SIZES, 2]

    # build grid anchors #######################################################
    num_ratios = len(level.anchor_aspect_ratios) * len(
        level.anchor_scale_ratios)
    grid_anchor_positions = grid_anchor_positions.reshape((*grid_size, 1,
                                                           2))  # [H, W, 1, 2]
    grid_anchor_positions = np.tile(grid_anchor_positions,
                                    (1, 1, num_ratios, 1))  # [H, W, SIZES, 2]

    grid_anchor_sizes = grid_anchor_sizes.reshape((1, 1, num_ratios,
                                                   2))  # [1, 1, SIZES, 2]
    grid_anchor_sizes = np.tile(grid_anchor_sizes,
                                (*grid_size, 1, 1))  # [H, W, SIZES, 2]

    grid_anchors = np.concatenate([grid_anchor_positions, grid_anchor_sizes],
                                  -1)  # [H, W, SIZES, 4]
    del grid_anchor_positions, grid_anchor_sizes

    # extract targets ##########################################################
    classes_true = np.array(
        [0] + [item.category_id for item in anns],
        dtype=np.float32)  # [OBJECTS]
    boxes_true = np.array(
        [[0.0, 0.0, 0.0, 0.0]] + [item.bbox for item in anns],
        dtype=np.float32)  # [OBJECTS, 4]
    assert classes_true.shape[0] == boxes_true.shape[0]

    # compute iou ##############################################################
    boxes_true = boxes_true.reshape(
        (boxes_true.shape[0], 1, 1, 1,
         boxes_true.shape[1]))  # [OBJECTS, 1, 1, 1, 4]

    grid_anchors = np.expand_dims(grid_anchors, 0)  # [1, H, W, SIZES, 4]

    iou = cv_utils.relative_iou(grid_anchors,
                                boxes_true)  # [OBJECTS, H, W, SIZES]
    iou *= iou > IOU_THRESHOLD

    # find best matches ########################################################
    indices = np.argmax(iou, 0)  # [H, W, SIZES]
    del iou
    assert indices.shape == (*grid_size, num_ratios)

    # build classification targets #############################################
    classification = classes_true[indices]  # [H, W, SIZES]
    assert classification.shape == (*grid_size, num_ratios)

    # build regression targets #################################################
    shifts = (boxes_true[..., :2] - grid_anchors[..., :2]
              ) / grid_anchors[..., 2:]  # [OBJECTS, H, W, SIZES, 2]
    scales = boxes_true[..., 2:] / grid_anchors[
        ..., 2:]  # [OBJECTS, H, W, SIZES, 2]
    scales = np.where(scales > 0, np.log(scales), scales * 0)
    shift_scales = np.concatenate([shifts, scales],
                                  -1)  # [OBJECTS, H, W, SIZES, 4]
    del shifts, scales

    # TODO: vectorize this
    indices_expanded = np.expand_dims(indices, -1)  # H * W * SIZES * 1
    del indices
    regression = np.zeros(
        (*grid_size, num_ratios, 4), dtype=np.float32)  # H * W * SIZES * 4
    for i in range(classes_true.shape[0]):
        regression += shift_scales[i] * (indices_expanded == i)

    return classification, regression


def make_labels(image, anns, levels, num_classes):
    labels = [
        make_level_labels(image, anns, l, num_classes=num_classes)
        for l in levels
    ]

    classifications, regressions = list(zip(*labels))

    return classifications, regressions


def gen(coco, dataset_path, levels, download):
    for img in coco.load_imgs(coco.get_img_ids()):
        filename = os.path.join(dataset_path, img.filename)
        anns = coco.load_anns(coco.get_ann_ids(img_ids=img.id))
        anns = list(anns)
        classifications, regressions = make_labels(
            img, anns, levels=levels, num_classes=coco.num_classes)

        yield filename.encode('utf-8'), classifications, regressions


def make_dataset(ann_path, dataset_path, levels, scale, shuffle, download):
    def make_gen():
        return gen(
            coco=coco,
            dataset_path=dataset_path,
            levels=levels,
            download=download)

    def load_image(filename):
        image = tf.read_file(filename)
        image = tf.image.decode_png(image, channels=3)
        image = tf.to_float(image) - 255 / 2

        return image

    def one_hot(classifications):
        return tuple(tf.one_hot(x, coco.num_classes) for x in classifications)

    def mapper(filename, classifications, regressions):
        def flip(image, classifications, regressions):
            image = tf.reverse(image, [1])
            classifications = tuple(
                tf.reverse(x, [1]) for x in classifications)
            regressions = tuple(tf.reverse(x, [1]) for x in regressions)
            regressions = tuple(
                tf.concat([x[..., :1], -x[..., 1:2], x[..., 2:]], -1)
                for x in regressions)

            return image, classifications, regressions

        image = load_image(filename)
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
        make_gen,
        output_types=(
            tf.string,
            tuple(tf.int32 for _ in levels),
            tuple(tf.float32 for _ in levels),
        ),
        output_shapes=(
            [],
            tuple([
                None, None,
                len(l.anchor_aspect_ratios) * len(l.anchor_scale_ratios)
            ] for l in levels),
            tuple([
                None, None,
                len(l.anchor_aspect_ratios) * len(l.anchor_scale_ratios), 4
            ] for l in levels),
        ))

    if shuffle is not None:
        ds = ds.shuffle(shuffle)
    ds = ds.map(mapper)

    return ds, coco.num_classes
