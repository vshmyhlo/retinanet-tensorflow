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

    # build grid anchor positions ##############################################
    # cell_size = image.size / grid_size
    # grid_y_positions = np.linspace(
    #     cell_size[0] / 2, image.size[0] - cell_size[0] / 2,
    #     grid_size[0]).reshape((grid_size[0], 1))  # [H, 1]
    # grid_x_positions = np.linspace(
    #     cell_size[1] / 2, image.size[1] - cell_size[1] / 2,
    #     grid_size[1]).reshape((1, grid_size[1]))  # [1, W]
    # del cell_size

    # grid_y_positions = np.tile(grid_y_positions, (1, grid_size[1]))  # [H, W]
    # grid_x_positions = np.tile(grid_x_positions, (grid_size[0], 1))  # [H, W]
    # assert grid_x_positions.shape == grid_y_positions.shape
    #
    # grid_anchor_positions = np.stack([grid_y_positions, grid_x_positions],
    #                                  -1)  # [H, W, 2]
    # del grid_x_positions, grid_y_positions
    #
    # # build grid anchor sizes ##################################################
    # grid_anchor_sizes = np.array(
    #     [
    #         compute_box_size(level.anchor_size, aspect_ratio, scale_ratio)
    #         for aspect_ratio, scale_ratio in level.anchor_aspect_scale_ratios
    #     ],
    #     dtype=np.float32)  # [SIZES, 2]
    #
    # # build grid anchors #######################################################
    # num_ratios = len(level.anchor_aspect_scale_ratios)
    # grid_anchor_positions = grid_anchor_positions.reshape((*grid_size, 1,
    #                                                        2))  # [H, W, 1, 2]
    # grid_anchor_positions = np.tile(grid_anchor_positions,
    #                                 (1, 1, num_ratios, 1))  # [H, W, SIZES, 2]
    #
    # grid_anchor_sizes = grid_anchor_sizes.reshape((1, 1, num_ratios,
    #                                                2))  # [1, 1, SIZES, 2]
    # grid_anchor_sizes = np.tile(grid_anchor_sizes,
    #                             (*grid_size, 1, 1))  # [H, W, SIZES, 2]
    #
    # anchor_box_grid = np.concatenate(
    #     [grid_anchor_positions, grid_anchor_sizes], -1)  # [H, W, SIZES, 4]
    # del grid_anchor_positions, grid_anchor_sizes
    #

    # extract targets ##########################################################

    # [OBJECTS]
    classes_true = tf.concat([[0], class_ids], 0)
    # [OBJECTS, 4]
    boxes_true = tf.concat([[[0, 0, 0, 0]], boxes], 0)
    boxes_true = tf.to_float(
        boxes_true / tf.concat([image_size, image_size], 0))

    # assert classes_true.shape[0] == boxes_true.shape[0]

    # compute iou ##############################################################

    # [OBJECTS, 1, 1, 1, 4]
    boxes_true_shape = tf.shape(boxes_true)
    boxes_true = tf.reshape(boxes_true, (boxes_true_shape[0], 1, 1, 1, 4))

    # anchor_box_grid = np.expand_dims(anchor_box_grid, 0)  # [1, H, W, SIZES, 4]

    # [1, H, W, SIZES, 4]
    anchor_boxmap = utils.anchor_boxmap(grid_size, anchor_boxes)

    # [OBJECTS, H, W, SIZES]
    iou = utils.iou(anchor_boxmap, boxes_true)
    iou = tf.where(iou > IOU_THRESHOLD, iou, tf.zeros_like(iou))
    # for the given anchor box, finds the ground truth box with the highest iou
    # [H, W, SIZES]
    indices = tf.argmax(iou, 0)
    del iou

    # assert indices.shape == (*grid_size, num_ratios)

    # build classification targets #############################################

    # [H, W, SIZES]
    classification = tf.gather(classes_true, indices)

    # assert classification.shape == (*grid_size, num_ratios)

    # build regression targets #################################################
    # shifts = (boxes_true[..., :2] - anchor_box_grid[..., :2]
    #           ) / anchor_box_grid[..., 2:]  # [OBJECTS, H, W, SIZES, 2]
    # scales = boxes_true[..., 2:] / anchor_box_grid[
    #     ..., 2:]  # [OBJECTS, H, W, SIZES, 2]
    # shift_scales = np.concatenate([shifts, scales],
    #                               -1)  # [OBJECTS, H, W, SIZES, 4]
    # del shifts, scales

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

    # [H, W, SIZES, 4]
    # anchor_boxmap_shape = tf.shape(anchor_boxmap)
    # regression = tf.zeros(anchor_boxmap_shape[1:], dtype=np.float32)

    # print(boxes_true)
    # print(indices_expanded)
    # fail
    # for i in range(classes_true.shape[0]):
    #     mask = tf.equal(indices_expanded, i)
    #     regression += boxes_true[i] * tf.to_float(mask)

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
