import tensorflow as tf
from data_loaders.inferred import Inferred
from dataset import build_dataset
from level import build_levels
import utils
import matplotlib.pyplot as plt


def draw_classmap(image, classification):
    fg_mask = utils.classmap_decode(classification).fg_mask
    fg_mask = tf.to_float(fg_mask)
    fg_mask = tf.expand_dims(fg_mask, -1)
    image_size = tf.shape(image)[:2]
    fg_mask = tf.image.resize_images(
        fg_mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

    image += fg_mask

    return image


def draw_mask(image, mask):
    mask = tf.to_float(mask)
    mask = tf.expand_dims(mask, -1)
    image_size = tf.shape(image)[:2]
    mask = tf.image.resize_images(
        mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

    image += mask

    return image


def draw_bounding_boxes(image, decoded, class_names):
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.py_func(
        lambda a, b, c, d: utils.draw_bounding_boxes(a, b, c, [x.decode() for x in d]),
        [image, decoded.boxes, decoded.class_ids, class_names],
        tf.uint8,
        stateful=False)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def main():
    levels = build_levels()
    data_loader = Inferred('shapes', ['./tmp', 10, 500])
    dataset = build_dataset(
        data_loader,
        levels=levels,
        scale=500,
        shuffle=4096,
        augment=True)

    input = dataset.make_one_shot_iterator().get_next()
    image = input['image']
    classifications = input['detection']['classifications']
    trainable_masks = input['trainable_masks']
    regressions_postprocessed = utils.dict_starmap(
        lambda r, l: utils.regression_postprocess(r, tf.to_float(l.anchor_sizes / input['image_size'])),
        [input['detection']['regressions'], levels])

    batch = []
    for i in range(image.shape[0]):
        level_images = {}
        for k in levels:
            class_row = []
            mask_row = []
            boxes_row = []

            for a in range(levels[k].anchor_sizes.shape[0]):
                image_with_classes = draw_classmap(image[i], classifications[k][i, :, :, a, :])
                image_with_classes /= tf.reduce_max(image_with_classes)
                class_row.append(image_with_classes)

                image_with_mask = draw_mask(image[i], trainable_masks[k][i, :, :, a])
                image_with_mask /= tf.reduce_max(image_with_mask)
                mask_row.append(image_with_mask)

                decoded = utils.boxes_decode(
                    classifications[k][i, :, :, a, :], regressions_postprocessed[k][i, :, :, a, :])

                image_with_boxes = draw_bounding_boxes(image[i], decoded, data_loader.class_names)
                image_with_boxes /= tf.reduce_max(image_with_boxes)
                boxes_row.append(image_with_boxes)

            class_row = tf.concat(class_row, 1)
            mask_row = tf.concat(mask_row, 1)
            boxes_row = tf.concat(boxes_row, 1)
            level_image = tf.concat([class_row, mask_row, boxes_row], 0)
            level_images[k] = level_image
        batch.append(level_images)

    with tf.Session() as sess:
        batch = sess.run(batch)
        for level_images in batch:
            for k in level_images:
                plt.figure(figsize=(16, 8))
                plt.imshow(level_images[k])
                plt.title(k)
                plt.show()


if __name__ == '__main__':
    main()
