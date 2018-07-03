import tensorflow as tf
from data_loaders.inferred import Inferred
from dataset import build_dataset
from level import build_levels
import utils
import matplotlib.pyplot as plt


def draw_classmap(image, classification):
    non_bg_mask = utils.classmap_decode(classification)['non_bg_mask']
    non_bg_mask = tf.to_float(non_bg_mask)
    non_bg_mask = tf.reduce_sum(non_bg_mask, -1)
    non_bg_mask = tf.expand_dims(non_bg_mask, -1)
    image_size = tf.shape(image)[:2]
    non_bg_mask = tf.image.resize_images(
        non_bg_mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

    image += non_bg_mask

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
    image_with_classmaps = utils.dict_map(lambda c: draw_classmap(image[0], c[0]), classifications)
    regressions_postprocessed = utils.dict_starmap(
        lambda r, l: utils.regression_postprocess(r, tf.to_float(l.anchor_sizes / input['image_size'])),
        [input['detection']['regressions'], levels])
    decoded = utils.dict_starmap(utils.boxes_decode, [classifications, regressions_postprocessed])

    with tf.Session() as sess:
        image, image_with_classmaps, decoded = sess.run([image, image_with_classmaps, decoded])
        assert image.shape[0] == 1

    for k in classifications:
        image_with_boxes = utils.draw_bounding_boxes(
            image[0], decoded[k].boxes, decoded[k].class_ids, data_loader.class_names)
        plt.subplot(121)
        plt.imshow(image_with_boxes)
        plt.subplot(122)
        plt.imshow(image_with_classmaps[k] / image_with_classmaps[k].max())

        plt.show()


if __name__ == '__main__':
    main()
