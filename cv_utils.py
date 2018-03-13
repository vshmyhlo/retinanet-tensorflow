import numpy as np


def to_absolute(box):
    center = box[..., :2]
    half_size = box[..., 2:] / 2
    box = np.concatenate([
        center - half_size,
        center + half_size,
    ], -1)

    return box


def relative_iou(box_a, box_b):
    return iou(to_absolute(box_a), to_absolute(box_b))


def iou(box_a, box_b):
    assert np.all(box_a[..., :2] <= box_a[..., 2:])  # TODO: should be <
    assert np.all(box_b[..., :2] <= box_b[..., 2:])  # TODO: should be <

    # determine the coordinates of the intersection rectangle
    y_top = np.maximum(box_a[..., 0], box_b[..., 0])
    x_left = np.maximum(box_a[..., 1], box_b[..., 1])
    y_bottom = np.minimum(box_a[..., 2], box_b[..., 2])
    x_right = np.minimum(box_a[..., 3], box_b[..., 3])

    invalid_mask = (y_bottom < y_top) + (x_right < x_left)

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
    iou = intersection_area / np.float32(
        box_a_area + box_b_area - intersection_area)
    print(iou)
    # TODO: zero division
    iou = np.where(invalid_mask, 0, iou)

    assert np.all(iou >= 0.0)
    assert np.all(iou <= 1.0)
    return iou
