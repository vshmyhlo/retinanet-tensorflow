import numpy as np

# def iou(box_a, box_b):
#   # determine the (x, y)-coordinates of the intersection rectangle
#   x_a = np.maximum(box_a[..., 0], box_b[..., 0])
#   y_a = np.maximum(box_a[..., 1], box_b[..., 1])
#   x_b = np.minimum(box_a[..., 2], box_b[..., 2])
#   y_b = np.minimum(box_a[..., 3], box_b[..., 3])
#
#   # compute the area of intersection rectangle
#   inter_area = (x_b - x_a + 1) * (y_b - y_a + 1)
#
#   # compute the area of both the prediction and ground-truth
#   # rectangles
#   box_a_area = (box_a[..., 2] - box_a[..., 0] + 1) * (box_a[..., 3] - box_a[..., 1] + 1)
#   box_b_area = (box_b[..., 2] - box_b[..., 0] + 1) * (box_b[..., 3] - box_b[..., 1] + 1)
#
#   # compute the intersection over union by taking the intersection
#   # area and dividing it by the sum of prediction + ground-truth
#   # areas - the interesection area
#   iou = inter_area / np.float32(box_a_area + box_b_area - inter_area)
#
#   # return the intersection over union value
#   return iou


def iou(box_a, box_b):
  assert np.all(box_a[..., :2] < box_a[..., 2:])
  assert np.all(box_b[..., :2] < box_b[..., 2:])

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
  iou = np.where(invalid_mask, 0, iou)

  assert np.all(iou >= 0.0)
  assert np.all(iou <= 1.0)
  return iou


if __name__ == '__main__':
  print(
      iou(
          np.array([
              [0.1, 0.1, 0.2, 0.2],
              [100, 100, 200, 200],
              [0.1, 0.1, 0.2, 0.2],
          ]),
          np.array([
              [0.1, 0.1, 0.3, 0.3],
              [100, 100, 300, 300],
              [100, 100, 300, 300],
          ])))
