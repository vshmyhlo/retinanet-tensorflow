import termcolor
import tensorflow as tf


def success(str):
    return termcolor.colored(str, 'green')


def warning(str):
    return termcolor.colored(str, 'yellow')


def danger(str):
    return termcolor.colored(str, 'red')


def log_args(args):
    print(warning('arguments:'))
    for key, value in sorted(vars(args).items(), key=lambda kv: kv[0]):
        print(warning('\t{}:').format(key), value)


def merge_outputs(tensors, name='merge_outputs'):
    with tf.name_scope(name):
        validate_shapes = [
            tf.assert_greater_equal(tf.rank(t), 4) for t in tensors
        ]
        with tf.control_dependencies(validate_shapes):
            reshaped = []
            for t in tensors:
                sh = tf.shape(t)
                sh = tf.concat([[sh[0], sh[1] * sh[2]], sh[3:]], 0)
                reshaped.append(tf.reshape(t, sh))

            return tf.concat(reshaped, 1)


def boxmap_anchor_relative_to_image_relative(regression):
    grid_size = tf.shape(regression)[1:3]
    cell_size = tf.to_float(1 / grid_size)

    y_pos = tf.linspace(cell_size[0] / 2, 1 - cell_size[0] / 2, grid_size[0])
    x_pos = tf.linspace(cell_size[1] / 2, 1 - cell_size[1] / 2, grid_size[1])

    x_pos, y_pos = tf.meshgrid(x_pos, y_pos)
    pos = tf.stack([y_pos, x_pos], -1)
    pos = tf.expand_dims(pos, -2)

    return tf.concat([regression[..., :2] + pos, regression[..., 2:]], -1)


def boxmap_center_relative_to_corner_relative(regression):
    pos = regression[..., :2]
    half_size = regression[..., 2:] / 2
    return tf.concat([pos - half_size, pos + half_size], -1)


def anchor_boxmap(grid_size, anchor_boxes):
    num_boxes = tf.shape(anchor_boxes)[0]
    positions = tf.zeros_like(anchor_boxes)
    anchor_boxes = tf.concat([positions, anchor_boxes], -1)
    anchor_boxes = tf.reshape(anchor_boxes, (1, 1, 1, num_boxes, 4))
    anchor_boxes = tf.tile(anchor_boxes, (1, grid_size[0], grid_size[1], 1, 1))

    boxmap = boxmap_anchor_relative_to_image_relative(anchor_boxes)
    boxmap = boxmap_center_relative_to_corner_relative(boxmap)

    return boxmap


def iou(box_a, box_b):
    # TODO: should be <
    with tf.control_dependencies([
            tf.assert_less_equal(box_a[..., :2], box_a[..., 2:]),
            tf.assert_less_equal(box_b[..., :2], box_b[..., 2:]),
    ]):
        # determine the coordinates of the intersection rectangle
        y_top = tf.maximum(box_a[..., 0], box_b[..., 0])
        x_left = tf.maximum(box_a[..., 1], box_b[..., 1])
        y_bottom = tf.minimum(box_a[..., 2], box_b[..., 2])
        x_right = tf.minimum(box_a[..., 3], box_b[..., 3])

    invalid_mask = tf.logical_or(y_bottom < y_top, x_right < x_left)

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (y_bottom - y_top) * (x_right - x_left)

    # compute the area of both AABBs
    box_a_area = (box_a[..., 2] - box_a[..., 0]) * (
        box_a[..., 3] - box_a[..., 1])
    box_b_area = (box_b[..., 2] - box_b[..., 0]) * (
        box_b[..., 3] - box_b[..., 1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / tf.to_float(
        box_a_area + box_b_area - intersection_area)
    iou = tf.where(invalid_mask, tf.zeros_like(iou), iou)

    with tf.control_dependencies([
            tf.assert_greater_equal(iou, 0.0),
            tf.assert_less_equal(iou, 1.0),
    ]):
        iou = tf.identity(iou)

    return iou
