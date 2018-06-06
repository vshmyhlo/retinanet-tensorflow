import termcolor
import tensorflow as tf
import cv2
import numpy as np


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


def boxmap_anchor_relative_to_image_relative(regression):
    grid_size = tf.shape(regression)[1:3]
    cell_size = tf.to_float(1 / grid_size)

    grid_y_pos = tf.linspace(cell_size[0] / 2, 1 - cell_size[0] / 2, grid_size[0])
    grid_x_pos = tf.linspace(cell_size[1] / 2, 1 - cell_size[1] / 2, grid_size[1])

    grid_x_pos, grid_y_pos = tf.meshgrid(grid_x_pos, grid_y_pos)
    grid_pos = tf.stack([grid_y_pos, grid_x_pos], -1)
    grid_pos = tf.expand_dims(grid_pos, -2)

    pos, size = tf.split(regression, 2, -1)

    return tf.concat([pos + grid_pos, size], -1)


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


def iou(a, b):
    # TODO: should be <
    with tf.control_dependencies([tf.assert_less_equal(a[..., :2], a[..., 2:]),
                                  tf.assert_less_equal(b[..., :2], b[..., 2:])]):
        # determine the coordinates of the intersection rectangle
        y_top = tf.maximum(a[..., 0], b[..., 0])
        x_left = tf.maximum(a[..., 1], b[..., 1])
        y_bottom = tf.minimum(a[..., 2], b[..., 2])
        x_right = tf.minimum(a[..., 3], b[..., 3])

    invalid_mask = tf.logical_or(y_bottom < y_top, x_right < x_left)

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (y_bottom - y_top) * (x_right - x_left)

    # compute the area of both AABBs
    box_a_area = (a[..., 2] - a[..., 0]) * (
            a[..., 3] - a[..., 1])
    box_b_area = (b[..., 2] - b[..., 0]) * (
            b[..., 3] - b[..., 1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / tf.to_float(
        box_a_area + box_b_area - intersection_area)
    iou = tf.where(invalid_mask, tf.zeros_like(iou), iou)

    with tf.control_dependencies([tf.assert_greater_equal(iou, 0.0), tf.assert_less_equal(iou, 1.0)]):
        iou = tf.identity(iou)

    return iou


def scale_regression(regression, anchor_boxes):
    anchor_boxes = tf.tile(anchor_boxes, (1, 2))
    anchor_boxes = tf.reshape(
        anchor_boxes, (1, 1, 1, anchor_boxes.shape[0], anchor_boxes.shape[1]))

    return regression * anchor_boxes


def regression_postprocess(regression, anchor_boxes, name='regression_postprocess'):
    with tf.name_scope(name):
        shifts, scales = tf.split(regression, 2, -1)
        regression = tf.concat([shifts, tf.exp(scales)], -1)

        regression = scale_regression(regression, anchor_boxes)
        regression = boxmap_anchor_relative_to_image_relative(regression)
        regression = boxmap_center_relative_to_corner_relative(regression)

        return regression


def draw_bounding_boxes(input, boxes, class_ids, class_names, num_classes):
    rng = np.random.RandomState(42)
    colors = [(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)) for _ in range(num_classes)]

    input_size = input.shape[:2]
    boxes_scale = np.array([*input_size, *input_size]) - 1  # TODO: -1 ?
    boxes = (boxes * boxes_scale).round().astype(np.int32)
    for box, class_id in zip(boxes, class_ids):
        input = cv2.rectangle(input, (box[1], box[0]), (box[3], box[2]), colors[class_id], 1)

        text_size, baseline = cv2.getTextSize(class_names[class_id], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        input = cv2.rectangle(
            input, (box[1], box[0] - text_size[1] - baseline), (box[1] + text_size[0], box[0]), colors[class_id], -1)
        input = cv2.putText(
            input, class_names[class_id], (box[1], box[0] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
            lineType=cv2.LINE_AA)

    return input


def classmap_decode(classmap):
    prob = tf.reduce_max(classmap, -1)
    non_bg = prob > 0.5
    classmap = tf.where(non_bg, tf.argmax(classmap, -1), tf.fill(tf.shape(non_bg), tf.to_int64(-1)))

    return classmap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image = cv2.imread('./data/tf-logo.png')
    image = cv2.resize(image, (400, 400))

    boxes = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.1, 0.1, 0.6, 0.6],
        [0.25, 0.25, 0.75, 0.75],
        [0.4, 0.4, 0.9, 0.9],
    ])

    class_ids = np.array([0, 3, 6, 9])

    class_names = list('abcdefghjk')

    image = draw_bounding_boxes(image, boxes, class_ids, class_names, 10)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
