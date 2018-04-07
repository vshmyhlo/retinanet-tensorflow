import os
import argparse
import itertools
import tensorflow as tf
# import tensorflow.contrib.slim as slim
import utils
import retinanet_v2 as retinanet
from level import build_levels
import objectives
import dataset
from tqdm import tqdm
import L4

# TODO: check regularization and initialization

# TODO: test network outputs scaling
# TODO: test session to evaluate
# TODO: check loss implementation
# TODO: check shuffle
# TODO: simplify architecture
# TODO: try focal cross-entropy
# TODO: check rounding and float32 conversions
# TODO: name_scope to variable_scope
# TODO: add dataset downloading to densenet
# TODO: exclude samples without prop IoU
# TODO: remove unnecessary validations
# TODO: set trainable parts
# TODO: flipping
# TODO: add augmentation
# TODO: boxes mapping should consider -1 index
# TODO: initialization


def print_summary(metrics, step):
    print(
        'step: {}, loss: {:.4f}, class_loss: {:.4f}, regr_loss: {:.4f}'.format(
            step, metrics['loss'], metrics['class_loss'],
            metrics['regr_loss']))


def heatmap_to_image(image, classification):
    image_size = tf.shape(image)[:2]
    heatmap = tf.argmax(classification, -1)
    heatmap = tf.reduce_max(heatmap, -1)
    heatmap = tf.not_equal(heatmap, 0)
    heatmap = tf.to_float(heatmap)
    heatmap = tf.expand_dims(heatmap, -1)
    heatmap = tf.image.resize_images(
        heatmap, image_size, method=tf.image.ResizeMethod.AREA)

    return heatmap


def draw_bounding_boxes(image,
                        regressions,
                        classifications,
                        max_output_size=1000):
    image = tf.expand_dims(image, 0)
    final_boxes = []
    final_scores = []

    for regression, classification in zip(regressions, classifications):
        mask = tf.not_equal(tf.argmax(classification, -1), 0)
        boxes = tf.boolean_mask(regression, mask)
        scores = tf.reduce_max(classification, -1)
        scores = tf.boolean_mask(scores, mask)

        final_boxes.append(boxes)
        final_scores.append(scores)

    final_boxes = tf.concat(final_boxes, 0)
    final_scores = tf.concat(final_scores, 0)
    nms_indices = tf.image.non_max_suppression(
        final_boxes, final_scores, max_output_size, iou_threshold=0.5)
    final_boxes = tf.expand_dims(tf.gather(final_boxes, nms_indices), 0)

    image = tf.image.draw_bounding_boxes(image, final_boxes)
    image = tf.squeeze(image, 0)

    return image


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, nargs=2, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--scale', type=int, default=600)
    parser.add_argument('--shuffle', type=int)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--clip-norm', type=float)
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['resnet', 'densenet'],
        default='densenet')
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['momentum', 'adam', 'l4'],
        default='momentum')

    return parser


def class_distribution(tensors):
    # TODO: do not average over batch
    return tf.stack([
        tf.reduce_mean(tf.to_float(tf.argmax(tensors[k], -1)), [0, 1, 2])
        for k in tensors
    ])


def make_train_step(loss, global_step, optimizer_type, learning_rate,
                    clip_norm):
    assert optimizer_type in ['momentum', 'adam', 'l4']

    if optimizer_type == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif optimizer_type == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer_type == 'l4':
        optimizer = L4.L4Adam(fraction=0.15)

    if clip_norm is None:
        # optimization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return optimizer.minimize(loss, global_step=global_step)
    else:
        # clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)

        # optimization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=global_step)


def make_metrics(class_loss, regr_loss, image, true, pred, level_names,
                 learning_rate):
    image = (image + 255 / 2) / 255
    classifications_true, regressions_true = true
    classifications_pred, regressions_pred = pred

    running_class_loss, update_class_loss = tf.metrics.mean(class_loss)
    running_regr_loss, update_regr_loss = tf.metrics.mean(regr_loss)
    running_true_class_dist, update_true_class_dist = tf.metrics.mean_tensor(
        class_distribution(classifications_true))
    running_pred_class_dist, update_pred_class_dist = tf.metrics.mean_tensor(
        class_distribution(classifications_pred))

    update_metrics = tf.group(update_class_loss, update_regr_loss,
                              update_true_class_dist, update_pred_class_dist)

    running_loss = running_class_loss + running_regr_loss

    metrics = {
        'loss': running_loss,
        'class_loss': running_class_loss,
        'regr_loss': running_regr_loss
    }

    running_summary = tf.summary.merge([
        tf.summary.scalar('class_loss', running_class_loss),
        tf.summary.scalar('regr_loss', running_regr_loss),
        tf.summary.scalar('loss', running_loss),
        tf.summary.scalar('learning_rate', learning_rate),
        tf.summary.histogram('classifications_true', running_true_class_dist),
        tf.summary.histogram('classifications_pred', running_pred_class_dist)
    ])

    image_summary = []

    for name, classifications, regressions in (
        ('true', classifications_true, regressions_true),
        ('pred', classifications_pred, regressions_pred),
    ):
        with tf.name_scope(name):
            image_with_boxes = draw_bounding_boxes(
                image[0], [regressions[pn][0] for pn in level_names],
                [classifications[pn][0] for pn in level_names])
            image_summary.append(
                tf.summary.image('boxmap', tf.expand_dims(image_with_boxes,
                                                          0)))

            heatmap_image = tf.zeros_like(image[0])
            for pn in level_names:
                heatmap_image += heatmap_to_image(image[0],
                                                  classifications[pn][0])

            heatmap_image = image[0] + heatmap_image
            image_summary.append(
                tf.summary.image('heatmap', tf.expand_dims(heatmap_image, 0)))

    image_summary = tf.summary.merge(image_summary)

    return metrics, update_metrics, running_summary, image_summary


def main():
    args = make_parser().parse_args()
    utils.log_args(args)

    levels = build_levels()
    training = tf.placeholder(tf.bool, [], name='training')
    global_step = tf.get_variable(
        'global_step', initializer=0, trainable=False)

    ds, num_classes = dataset.make_dataset(
        ann_path=args.dataset[0],
        dataset_path=args.dataset[1],
        levels=levels,
        scale=args.scale,
        shuffle=args.shuffle,
        download=False,
        augment=True)

    iter = ds.make_initializable_iterator()
    image, classifications_true, regressions_true = iter.get_next()

    net = retinanet.RetinaNet(
        levels=levels,
        num_classes=num_classes,
        dropout_rate=args.dropout,
        backbone=args.backbone,
        # weight_decay=args.weight_decay,
    )
    classifications_pred, regressions_pred = net(image, training)

    class_loss, regr_loss = objectives.loss(
        (classifications_true, regressions_true),
        (classifications_pred, regressions_pred))

    loss = class_loss + regr_loss
    train_step = make_train_step(
        loss,
        global_step=global_step,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        clip_norm=args.clip_norm)

    metrics, update_metrics, running_summary, image_summary = make_metrics(
        class_loss,
        regr_loss,
        image=image,
        true=(classifications_true, regressions_true),
        pred=(classifications_pred, regressions_pred),
        level_names=levels.keys(),
        learning_rate=args.learning_rate)

    globals_init = tf.global_variables_initializer()
    locals_init = tf.local_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess, tf.summary.FileWriter(
            logdir=os.path.join(args.experiment, 'train'),
            graph=sess.graph) as train_writer:
        restore_path = tf.train.latest_checkpoint(args.experiment)
        if restore_path:
            saver.restore(sess, restore_path)
        else:
            sess.run(globals_init)

        for epoch in range(args.epochs):
            sess.run([iter.initializer, locals_init])

            for _ in tqdm(itertools.count()):
                try:
                    _, step = sess.run(
                        [(train_step, update_metrics), global_step], {
                            training: True
                        })

                    if step % args.log_interval == 0:
                        m, run_summ, img_summ = sess.run(
                            [metrics, running_summary, image_summary], {
                                training: True
                            })

                        print()
                        print_summary(m, step)
                        train_writer.add_summary(run_summ, step)
                        train_writer.add_summary(img_summ, step)
                        train_writer.flush()
                        saver.save(sess,
                                   os.path.join(args.experiment, 'model.ckpt'))
                        sess.run(locals_init)
                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    main()
