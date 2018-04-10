import os
import numpy as np
import tensorflow as tf
from coco import COCO
import utils
import augmentation
import argparse
import itertools
from tqdm import tqdm

IOU_THRESHOLD = 0.5
MEAN = [0.46618041, 0.44669811, 0.40252436]
STD = [0.27940595, 0.27489075, 0.28920765]


# TODO: remove image size and make all boxes 0-1
# TODO: background category
# TODO: ignored boxes
def level_labels(image_size, class_ids, boxes, level, factor):
    grid_size = tf.to_int32(tf.ceil(image_size / factor))
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
    labels = {
        pn: level_labels(
            image_size,
            class_ids,
            boxes,
            level=levels[pn],
            factor=2**int(pn[-1]))
        for pn in levels
    }

    classifications = {pn: labels[pn][0] for pn in labels}
    regressions = {pn: labels[pn][1] for pn in labels}

    return classifications, regressions


def gen(coco):
    for img in coco.load_imgs(coco.get_img_ids()):
        filename = os.path.join(coco.dataset_path,
                                img.filename).encode('utf-8')
        anns = list(coco.load_anns(coco.get_ann_ids(img_ids=img.id)))
        class_ids = np.array([a.category_id for a in anns])
        boxes = np.array([a.box for a in anns])

        # TODO: check why dataset has samples without boxes
        if len(anns) > 0:
            yield filename, class_ids, boxes
        else:
            yield filename, np.zeros([0]), np.zeros([0, 4])


def rescale_image(image, scale):
    size = tf.to_float(tf.shape(image)[:2])
    shorter = tf.argmin(size)
    ratio = scale / size[shorter]
    new_size = tf.to_int32(tf.round(size * ratio))
    new_size = tf.Print(new_size, [new_size])

    return tf.image.resize_images(image, new_size, method=tf.image.ResizeMethod.BILINEAR)


def make_dataset(ann_path,
                 dataset_path,
                 levels,
                 download,
                 augment,
                 num_threads=os.cpu_count() // 2,
                 scale=None,
                 shuffle=None):
    def load_image_with_labels(filename, class_ids, boxes):
        def load_image(filename):
            image = tf.read_file(filename)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)

            return image

        image = load_image(filename)

        if scale is not None:
            image = rescale_image(image, scale)

        image_size = tf.shape(image)[:2]
        classifications, regressions = make_labels(
            image_size, class_ids, boxes, levels=levels)

        return image, classifications, regressions

    def preprocess(image, classifications, regressions):
        image_flipped, classifications_flipped, regressions_flipped = augmentation.flip(
            image, classifications, regressions)

        image = tf.stack([image, image_flipped], 0)
        classifications = {
            pn: tf.stack([classifications[pn], classifications_flipped[pn]], 0)
            for pn in classifications
        }
        regressions = {
            pn: tf.stack([regressions[pn], regressions_flipped[pn]], 0)
            for pn in regressions
        }
        # TODO: use level names
        classifications = {
            pn: tf.one_hot(classifications[pn], coco.num_classes)
            for pn in classifications
        }

        return image, classifications, regressions

    def augment_sample(image, classifications, regressions):
        # TODO: add augmentation
        # image = tf.image.random_contrast(image, 0.8, 1.2)
        # image = tf.image.random_brightness(image, 0.2)
        # image = tf.image.random_saturation(image, 0.8, 1.0)

        return image, classifications, regressions

    coco = COCO(ann_path, dataset_path, download)
    ds = tf.data.Dataset.from_generator(
        lambda: gen(coco),
        output_types=(tf.string, tf.int32, tf.int32),
        output_shapes=([], [None], [None, 4]))

    if shuffle is not None:
        ds = ds.shuffle(shuffle)

    ds = ds.map(load_image_with_labels, num_parallel_calls=num_threads)
    ds = ds.map(preprocess, num_parallel_calls=num_threads)

    if augment:
        ds = ds.map(augment_sample, num_parallel_calls=num_threads)

    return ds, coco.num_classes


def compute_mean_std():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs=2, required=True)

    args = parser.parse_args()
    ds, num_classes = make_dataset(
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
