import dataset
from level import build_levels
import tensorflow as tf
from tqdm import tqdm
import os
from train import draw_bounding_boxes, draw_classmap
import matplotlib.pyplot as plt
import utils
import itertools

if __name__ == '__main__':
    levels = build_levels()

    ds = dataset.build_dataset(
        spec=['pascal', os.path.expanduser('~/Datasets/pascal/VOCdevkit/VOC2012'), 'trainval'],
        levels=levels,
        # scale=350,
        scale=600,
        augment=True)

    iter = ds['dataset'].prefetch(1).make_one_shot_iterator()
    input = iter.get_next()
    image_size = tf.shape(input['image'])[1:3]

    input['regressions'] = {
        pn: utils.regression_postprocess(
            input['regressions'][pn], tf.to_float(levels[pn].anchor_sizes / image_size)) for pn
        in input['regressions']}

    image_with_classmap = draw_classmap(
        input['image'][0],
        {pn: input['classifications'][pn][0] for pn in input['classifications']})
    image_with_boxes, final_boxes = draw_bounding_boxes(
        input['image'][0],
        {pn: input['classifications'][pn][0] for pn in input['classifications']},
        {pn: input['regressions'][pn][0] for pn in input['regressions']})

    image_with_boxes_true = input['image'][:1]

    nms_indices = tf.image.non_max_suppression(input['boxes'], tf.ones(tf.shape(input['boxes'])[:1]), 1000)
    boxes = tf.gather(input['boxes'], nms_indices)
    image_with_boxes_true = tf.image.draw_bounding_boxes(image_with_boxes_true, [boxes])
    image_with_boxes_true = tf.squeeze(image_with_boxes_true, 0)

    with tf.Session() as sess:
        i = 0

        for _ in tqdm(itertools.count()):
            tb, pb, ic, ib, it = sess.run(
                [boxes, final_boxes, image_with_classmap, image_with_boxes, image_with_boxes_true])

            if tb.shape != pb.shape:
                print()
                print('tb.shape: {.shape}, pb.shape: {.shape}'.format(tb, pb))

                plt.figure(figsize=(16, 8))
                plt.subplot(2, 2, 1)
                plt.axis('off')
                plt.imshow(ic / ic.max())
                plt.subplot(2, 2, 3)
                plt.axis('off')
                plt.imshow(ib / ib.max())
                plt.subplot(2, 2, 4)
                plt.axis('off')
                plt.imshow(it / it.max())

                plt.tight_layout()
                plt.show()

                i += 1

                if i == 8:
                    break
