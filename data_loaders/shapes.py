from data_loaders.base import Base
import cv2
import os


# TODO: refactor

class Shapes(Base):
    def __init__(self, path, num_samples, image_size):
        self._path = path
        self._num_samples = num_samples
        self._image_size = image_size
        self._class_names = ['square', 'triangle', 'circle']

    @property
    def class_names(self):
        return self._class_names

    @property
    def num_classes(self):
        return len(self._class_names)

    def __iter__(self):
        if not os.path.exists(self._path):
            os.mkdir(self._path)

        for i in range(self._num_samples):
            bg_color, shapes = random_image(self._image_size)
            bg_color = np.array(bg_color).reshape([1, 1, 3])
            image = np.ones([*self._image_size, 3], dtype=np.uint8)
            image = image * bg_color.astype(np.uint8)
            boxes = []
            class_ids = []
            for shape, color, dims in shapes:
                image = draw_shape(image, shape, dims, color)
                x, y, s = dims
                boxes.append([
                    (y - s) / self._image_size[0],
                    (x - s) / self._image_size[1],
                    (y + s) / self._image_size[0],
                    (x + s) / self._image_size[1]
                ])
                class_ids.append(shape)

            boxes = np.array(boxes)
            class_ids = np.array([self._class_names.index(class_id) for class_id in class_ids])

            image_file = os.path.join(self._path, '{}.png'.format(i))
            cv2.imwrite(image_file, image)

            yield {
                'image_file': image_file.encode('utf-8'),
                'class_ids': class_ids,
                'boxes': boxes
            }


import numpy as np
import random
import cv2
import math
import matplotlib.pyplot as plt
from itertools import islice, count
import tensorflow as tf


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def draw_shape(image, shape, dims, color):
    """Draws a shape from the given specs."""
    # Get the center x, y and the size s
    x, y, s = dims
    if shape == 'square':
        cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
    elif shape == "circle":
        cv2.circle(image, (x, y), s, color, -1)
    elif shape == "triangle":
        points = np.array(
            [[
                (x, y - s),
                (x - s / math.sin(math.radians(60)), y + s),
                (x + s / math.sin(math.radians(60)), y + s),
            ]],
            dtype=np.int32)
        cv2.fillPoly(image, points, color)
    return image


def random_shape(image_size):
    """Generates specifications of a random shape that lies within
    the given height and width boundaries.
    Returns a tuple of three valus:
    * The shape name (square, circle, ...)
    * Shape color: a tuple of 3 values, RGB.
    * Shape dimensions: A tuple of values that define the shape size
                        and location. Differs per shape type.
    """
    # Shape
    shape = random.choice(["square", "circle", "triangle"])
    # Color
    color = tuple([random.randint(0, 255) for _ in range(3)])
    # Center x, y
    buffer = 20
    y = random.randint(buffer, image_size[0] - buffer - 1)
    x = random.randint(buffer, image_size[1] - buffer - 1)
    # Size
    s = random.randint(buffer, image_size[0] // 4)
    return shape, color, (x, y, s)


def random_image(image_size):
    """Creates random specifications of an image with multiple shapes.
    Returns the background color of the image and a list of shape
    specifications that can be used to draw the image.
    """
    # Pick random background color
    bg_color = np.array([random.randint(0, 255) for _ in range(3)])
    # Generate a few random shapes and record their
    # bounding boxes
    shapes = []
    boxes = []
    N = random.randint(1, 4)
    for _ in range(N):
        shape, color, dims = random_shape(image_size)
        shapes.append((shape, color, dims))
        x, y, s = dims
        boxes.append([y - s, x - s, y + s, x + s])
    # Apply non-max suppression wit 0.3 threshold to avoid
    # shapes covering each other
    keep_ixs = non_max_suppression(np.array(boxes), np.arange(N), 0.3)
    shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
    return bg_color, shapes


# def generator(image_size):


def make_dataset(levels, batch_size, image_size):
    def preprocess(image, class_id, boxes):
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, class_id, boxes

    ds = tf.data.Dataset.from_generator(
        lambda: generator(image_size),
        output_types=(tf.int32, tf.int32, tf.int32),
        output_shapes=((*image_size, 3), (), (None, 4)))
    ds = ds.map(preprocess)
    ds = ds.batch(batch_size)

    return ds, len(shape_to_id)


def main():
    dl = Shapes(600, 600)
    image_size = 256, 256

    for image, class_id, boxes in islice(generator(image_size), 5):
        for box in boxes:
            cv2.rectangle(
                image,
                (box[1], box[0]),
                (box[3], box[2]),
                (0, 255, 0),
            )
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    main()
