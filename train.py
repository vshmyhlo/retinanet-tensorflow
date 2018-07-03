import os
import argparse
import itertools
import tensorflow as tf
import utils
import retinanet
from level import build_levels
import losses
import dataset
from tqdm import tqdm
import L4
from data_loaders.inferred import Inferred


# TODO: check fpn relu activation usage
# TODO: rename non_bg to fg
# TODO: typing
# TODO: check retinanet encodes background as 0 everywhere
# TODO: compute only loss on train
# TODO: estimator
# TODO: classwise nms
# TODO: check preprocessing
# TODO: optimize data loading
# TODO: add correct dropout everywhere
# TODO: dropout noise shape
# TODO: check classmap decoded uses scaled logits (sigmoid)

# TODO: dict map vs dict starmap

# TODO: move label creation to graph
# TODO: check focal-cross-entropy
# TODO: try focal cross-entropy
# TODO: anchor assignment
# TODO: check rounding and float32 conversions
# TODO: add dataset downloading to densenet
# TODO: exclude samples without prop IoU
# TODO: set trainable parts
# TODO: use trainable_mask for visualization
# TODO: check if batch norm after dropout is ok
# TODO: balances cross-entropy
# TODO: why sometimes ground true boxes not drawn
# TODO: class iou
# TODO: regr iou

def preprocess_image(image):
    return (image - dataset.MEAN) / dataset.STD


def print_summary(metrics, step):
    print(
        'step: {}, total_loss: {:.4f}, class_loss: {:.4f}, regr_loss: {:.4f}, regularization_loss: {:.4f}'.format(
            step, metrics['total_loss'], metrics['class_loss'], metrics['regr_loss'], metrics['regularization_loss']))


def cyclical_learning_rate(min, max, step_size, global_step):
    cycle_size = step_size * 2
    step = global_step % cycle_size
    k = tf.cond(step < step_size, lambda: step / step_size, lambda: 1 - (step - step_size) / step_size)
    learning_rate = min + (max - min) * k

    return learning_rate


def draw_classmap(image, classifications):
    for k in classifications:
        classification = classifications[k]
        non_bg_mask = utils.classmap_decode(classification)['non_bg_mask']
        non_bg_mask = tf.to_float(non_bg_mask)
        non_bg_mask = tf.reduce_sum(non_bg_mask, -1)
        non_bg_mask = tf.expand_dims(non_bg_mask, -1)
        image_size = tf.shape(image)[:2]
        non_bg_mask = tf.image.resize_images(
            non_bg_mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

        image += non_bg_mask

    return image


def draw_bounding_boxes(image, classifications, regressions, class_names):
    decoded = []

    for k in classifications:
        decoded.append(utils.boxes_decode(classifications[k], regressions[k]))

    decoded = utils.merge_boxes_decoded(decoded)
    decoded = utils.nms_classwise(decoded, num_classes=len(class_names))

    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.py_func(
        lambda a, b, c, d: utils.draw_bounding_boxes(a, b, c, [x.decode() for x in d]),
        [image, decoded.boxes, decoded.class_ids, class_names],
        tf.uint8,
        stateful=False)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, nargs='+', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--scale', type=int, default=600)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--grad-clip-norm', type=float)
    parser.add_argument('--focal-loss-alpha', type=float, default=0.25)
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['resnet', 'densenet_121', 'densenet_169', 'mobilenet_v2'],
        default='resnet')
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['momentum', 'adam', 'l4'],
        default='momentum')

    return parser


def build_train_step(loss, learning_rate, global_step, config):
    assert config.optimizer in ['momentum', 'adam', 'l4']

    if config.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif config.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif config.optimizer == 'l4':
        optimizer = L4.L4Adam(fraction=0.15)
    else:
        raise AssertionError('invalid optimizer type: {}'.format(config.optimizer))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if config.grad_clip_norm is not None:
            grads_and_vars = optimizer.compute_gradients(loss)
            grads = [x[0] for x in grads_and_vars]
            vars = [x[1] for x in grads_and_vars]
            grads, _ = tf.clip_by_global_norm(grads, config.grad_clip_norm)
            return optimizer.apply_gradients(zip(grads, vars), global_step=global_step)
        else:
            return optimizer.minimize(loss, global_step=global_step)


def build_metrics(total_loss, class_loss, regr_loss, regularization_loss, labels, logits):
    # def build_iou(labels, logits, name='build_iou'):
    #     with tf.name_scope(name):
    #         # decode both using ground true classification
    #         labels_decoded = utils.boxes_decode(labels['classifications'], labels['regressions_postprocessed'])
    #         logits_decoded = utils.boxes_decode(labels['classifications'], logits['regressions_postprocessed'])
    #         return utils.iou(labels_decoded.boxes, logits_decoded.boxes)

    metrics = {}
    update_metrics = {}

    # TODO: refactor
    # TODO: fix iou to use class ids
    # metrics['class_iou'], update_metrics['class_iou'] = tf.metrics.mean_iou(
    #     labels=labels['detection_trainable']['classifications'],
    #     predictions=tf.to_int32(tf.nn.sigmoid(logits['detection_trainable']['classifications']) > 0.5),
    #     num_classes=2)
    # metrics['regr_iou'], update_metrics['regr_iou'] = tf.metrics.mean(
    #     build_iou(labels['detection_trainable'], logits['detection_trainable']))
    metrics['total_loss'], update_metrics['total_loss'] = tf.metrics.mean(total_loss)
    metrics['class_loss'], update_metrics['class_loss'] = tf.metrics.mean(class_loss)
    metrics['regr_loss'], update_metrics['regr_loss'] = tf.metrics.mean(regr_loss)
    metrics['regularization_loss'], update_metrics['regularization_loss'] = tf.metrics.mean(regularization_loss)

    return metrics, update_metrics


def build_summary(metrics, image, labels, logits, learning_rate, class_names):
    summary = [
        # tf.summary.scalar('class_iou', metrics['class_iou']),
        # tf.summary.scalar('regr_iou', metrics['regr_iou']),
        tf.summary.scalar('total_loss', metrics['total_loss']),
        tf.summary.scalar('class_loss', metrics['class_loss']),
        tf.summary.scalar('regr_loss', metrics['regr_loss']),
        tf.summary.scalar('regularization_loss', metrics['regularization_loss']),
        tf.summary.scalar('learning_rate', learning_rate),
    ]

    image = image * dataset.STD + dataset.MEAN
    # TODO: better scope names
    for scope, classifications, regressions in (
            ('true',
             labels['detection'].classification.prob,
             labels['detection'].regression_postprocessed),
            ('pred',
             logits['detection'].classification.prob,
             logits['detection'].regression_postprocessed)
    ):
        for i in range(image.shape[0]):
            with tf.name_scope('{}/{}'.format(scope, i)):
                image_with_boxes = draw_bounding_boxes(
                    image[i],
                    utils.dict_map(lambda x: x[i], classifications),
                    utils.dict_map(lambda x: x[i], regressions),
                    class_names=class_names)
                summary.append(tf.summary.image('regression', tf.expand_dims(image_with_boxes, 0)))

                image_with_classmap = draw_classmap(
                    image[i], utils.dict_map(lambda x: x[i], classifications))
                summary.append(tf.summary.image('classification', tf.expand_dims(image_with_classmap, 0)))

    # summary = tf.summary.merge(summary)
    summary = tf.summary.merge_all()  # FIXME:

    return summary


def build_learning_rate(global_step, config):
    # return cyclical_learning_rate(1e-3, 3., 5000, global_step)
    return config.learning_rate


def main():
    args = build_parser().parse_args()
    utils.log_args(args)

    levels = build_levels()
    training = tf.placeholder(tf.bool, [], name='training')
    global_step = tf.get_variable('global_step', initializer=0, trainable=False)

    data_loader = Inferred(args.dataset[0], args.dataset[1:])
    ds = dataset.build_dataset(
        data_loader,
        levels=levels,
        scale=args.scale,
        shuffle=4096,
        augment=True)

    iter = ds.prefetch(1).make_initializable_iterator()
    input = iter.get_next()
    input = {**input, 'image': preprocess_image(input['image'])}

    net = retinanet.RetinaNet(
        levels=levels,
        num_classes=data_loader.num_classes,
        activation=tf.nn.elu,
        dropout_rate=args.dropout,
        backbone=args.backbone)
    logits = {'detection': net(input['image'], training)}
    input, logits = utils.process_labels_and_logits(labels=input, logits=logits, levels=levels)

    class_loss, regr_loss = losses.loss(
        labels=input['detection_trainable'],
        logits=logits['detection_trainable'],
        class_loss_kwargs={'alpha': args.focal_loss_alpha})
    regularization_loss = tf.losses.get_regularization_loss()

    total_loss = class_loss + regr_loss + regularization_loss
    learning_rate = build_learning_rate(global_step, args)
    train_step = build_train_step(total_loss, learning_rate, global_step=global_step, config=args)

    metrics, update_metrics = build_metrics(
        total_loss,
        class_loss,
        regr_loss,
        regularization_loss,
        labels=input,
        logits=logits)

    summary = build_summary(
        metrics,
        image=input['image'],
        labels=input,
        logits=logits,
        learning_rate=learning_rate,
        class_names=data_loader.class_names)

    globals_init = tf.global_variables_initializer()
    locals_init = tf.local_variables_initializer()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess, tf.summary.FileWriter(
            logdir=os.path.join(args.experiment, 'train')) as train_writer:
        restore_path = tf.train.latest_checkpoint(args.experiment)
        if restore_path:
            saver.restore(sess, restore_path)
            print('model restored from {}'.format(restore_path))
        else:
            sess.run(globals_init)

        for epoch in range(args.epochs):
            sess.run([iter.initializer, locals_init])

            for _ in tqdm(itertools.count()):
                try:
                    _, step = sess.run(
                        [(train_step, update_metrics), global_step], {training: True})

                    if args.log_interval is not None and step % args.log_interval == 0:
                        m, s = sess.run(
                            [metrics, summary], {training: True})

                        print()
                        print_summary(m, step)
                        train_writer.add_summary(s, step)
                        saver.save(sess, os.path.join(args.experiment, 'model.ckpt'), write_meta_graph=False)
                        sess.run(locals_init)
                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    main()
