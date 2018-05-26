import os
import argparse
import itertools
import tensorflow as tf
import utils
import retinanet
from level import build_levels
import objectives
import dataset
from tqdm import tqdm
import L4


# TODO: move label creation to graph
# TODO: check focal-cross-entropy
# TODO: try focal cross-entropy
# TODO: anchor assignment
# TODO: check rounding and float32 conversions
# TODO: add dataset downloading to densenet
# TODO: exclude samples without prop IoU
# TODO: set trainable parts
# TODO: use not_ignored_mask for visualization
# TODO: check if batch norm after dropout is ok
# TODO: balances cross-entropy
# TODO: try not using bg class
# TODO: align corners in interpolation

def preprocess_image(image):
    return (image - dataset.MEAN) / dataset.STD


def print_summary(metrics, step):
    print(
        'step: {}, loss: {:.4f}, class_loss: {:.4f}, regr_loss: {:.4f}, regularization_loss: {:.4f}'.format(
            step, metrics['loss'], metrics['class_loss'], metrics['regr_loss'], metrics['regularization_loss']))


def classmap_to_image(image, classmap):
    image_size = tf.shape(image)[:2]
    classmap = tf.reduce_max(classmap, -1)
    classmap = tf.not_equal(classmap, 0)
    classmap = tf.to_float(classmap)
    classmap = tf.expand_dims(classmap, -1)
    classmap = tf.image.resize_images(classmap, image_size, method=tf.image.ResizeMethod.AREA, align_corners=True)

    return classmap


def draw_bounding_boxes(image, regressions, classifications, max_output_size=1000):
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
    nms_indices = tf.image.non_max_suppression(final_boxes, final_scores, max_output_size, iou_threshold=0.5)
    final_boxes = tf.gather(final_boxes, nms_indices)
    final_boxes = tf.expand_dims(final_boxes, 0)

    image = tf.image.draw_bounding_boxes(image, final_boxes)
    image = tf.squeeze(image, 0)

    return image


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, nargs=2, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--scale', type=int, default=600)
    parser.add_argument('--experiment', type=str, required=True)
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


def make_train_step(loss, global_step, optimizer_type, learning_rate):
    assert optimizer_type in ['momentum', 'adam', 'l4']

    if optimizer_type == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif optimizer_type == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer_type == 'l4':
        optimizer = L4.L4Adam(fraction=0.15)

    # optimization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        return optimizer.minimize(loss, global_step=global_step)


def make_metrics(loss, class_loss, regr_loss, regularization_loss, image, true, pred, not_ignored_masks, levels,
                 learning_rate):
    image_size = tf.shape(image)[1:3]
    image = image * dataset.STD + dataset.MEAN
    classifications_true, regressions_true = true
    classifications_pred, regressions_pred = pred
    regressions_true = {
        pn: utils.regression_postprocess(regressions_true[pn], tf.to_float(levels[pn].anchor_sizes / image_size)) for
        pn in regressions_true}
    regressions_pred = {
        pn: utils.regression_postprocess(regressions_pred[pn], tf.to_float(levels[pn].anchor_sizes / image_size)) for
        pn in regressions_pred}

    metrics = {}
    update_metrics = {}

    metrics['loss'], update_metrics['loss'] = tf.metrics.mean(loss)
    metrics['class_loss'], update_metrics['class_loss'] = tf.metrics.mean(class_loss)
    metrics['regr_loss'], update_metrics['regr_loss'] = tf.metrics.mean(regr_loss)
    metrics['regularization_loss'], update_metrics['regularization_loss'] = tf.metrics.mean(regularization_loss)
    # running_true_class_dist, update_true_class_dist = tf.metrics.mean_tensor(
    #     class_distribution(classifications_true))
    # running_pred_class_dist, update_pred_class_dist = tf.metrics.mean_tensor(
    #     class_distribution(classifications_pred))

    running_summary = tf.summary.merge([
        tf.summary.scalar('loss', metrics['loss']),
        tf.summary.scalar('class_loss', metrics['class_loss']),
        tf.summary.scalar('regr_loss', metrics['regr_loss']),
        tf.summary.scalar('regularization_loss', metrics['regularization_loss']),
        tf.summary.scalar('learning_rate', learning_rate),
        # tf.summary.histogram('classifications_true', running_true_class_dist),
        # tf.summary.histogram('classifications_pred', running_pred_class_dist)
    ])

    image_summary = []

    # TODO: better scope names
    for name, classifications, regressions in (
            ('true', classifications_true, regressions_true),
            ('pred', classifications_pred, regressions_pred),
    ):
        for i in range(image.shape[0]):
            with tf.name_scope('{}/{}'.format(name, i)):
                image_with_boxes = draw_bounding_boxes(
                    image[i], [regressions[pn][i] for pn in regressions],
                    [classifications[pn][i] for pn in classifications])
                image_summary.append(tf.summary.image('regression', tf.expand_dims(image_with_boxes, 0)))

                classmap_image = tf.zeros_like(image[i])
                for pn in classifications:
                    classmap_image += classmap_to_image(image[i], tf.argmax(classifications[pn][i], -1))
                classmap_image = image[i] + classmap_image
                image_summary.append(tf.summary.image('classification', tf.expand_dims(classmap_image, 0)))

    # for i in range(image.shape[0]):
    #     with tf.name_scope('{}'.format(name, i)):
    #         not_ignored_mask_image = tf.zeros_like(image[i])
    #         for pn in not_ignored_masks:
    #             not_ignored_mask_image += classmap_to_image(image[i], tf.to_float(not_ignored_masks[pn][i]))
    #         not_ignored_mask_image = image[i] + not_ignored_mask_image
    #         image_summary.append(tf.summary.image('not_ignored_mask', tf.expand_dims(not_ignored_mask_image, 0)))

    image_summary = tf.summary.merge(image_summary)

    return metrics, update_metrics, running_summary, image_summary


def main():
    args = make_parser().parse_args()
    utils.log_args(args)

    levels = build_levels()
    training = tf.placeholder(tf.bool, [], name='training')
    global_step = tf.get_variable('global_step', initializer=0, trainable=False)

    ds, num_classes = dataset.make_dataset(
        ann_path=args.dataset[0],
        dataset_path=args.dataset[1],
        levels=levels,
        scale=args.scale,
        download=False,
        augment=True)

    iter = ds.shuffle(32).prefetch(1).make_initializable_iterator()
    input = iter.get_next()
    input = {
        **input,
        'image': preprocess_image(input['image'])
    }

    net = retinanet.RetinaNet(
        levels=levels,
        num_classes=num_classes,
        dropout_rate=args.dropout,
        backbone=args.backbone)
    classifications_pred, regressions_pred = net(input['image'], training)
    assert input['classifications'].keys() == input['regressions'].keys() == levels.keys()
    assert classifications_pred.keys() == regressions_pred.keys() == levels.keys()

    class_loss, regr_loss = objectives.loss(
        (input['classifications'], input['regressions']),
        (classifications_pred, regressions_pred),
        not_ignored_masks=input['not_ignored_masks'])
    regularization_loss = tf.losses.get_regularization_loss()

    loss = class_loss + regr_loss + regularization_loss
    train_step = make_train_step(
        loss,
        global_step=global_step,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate)

    metrics, update_metrics, running_summary, image_summary = make_metrics(
        loss,
        class_loss,
        regr_loss,
        regularization_loss,
        image=input['image'],
        true=(input['classifications'], input['regressions']),
        pred=(classifications_pred, regressions_pred),
        not_ignored_masks=input['not_ignored_masks'],
        levels=levels,
        learning_rate=args.learning_rate)

    globals_init = tf.global_variables_initializer()
    locals_init = tf.local_variables_initializer()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess, tf.summary.FileWriter(
            logdir=os.path.join(args.experiment, 'train'), graph=sess.graph) as train_writer:
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
                        saver.save(sess,
                                   os.path.join(args.experiment, 'model.ckpt'))
                        sess.run(locals_init)
                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    main()
