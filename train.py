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


def draw_bounding_boxes(image, classifications, regressions, max_output_size=1000):
    final_boxes = []
    final_scores = []

    for k in classifications:
        decoded = utils.boxes_decode(classifications[k], regressions[k])
        final_boxes.append(decoded['boxes'])
        final_scores.append(decoded['scores'])

    final_boxes = tf.concat(final_boxes, 0)
    final_scores = tf.concat(final_scores, 0)
    nms_indices = tf.image.non_max_suppression(final_boxes, final_scores, max_output_size, iou_threshold=0.5)
    final_boxes = tf.gather(final_boxes, nms_indices)
    final_boxes = tf.expand_dims(final_boxes, 0)

    image = tf.expand_dims(image, 0)
    image = tf.image.draw_bounding_boxes(image, final_boxes)
    image = tf.squeeze(image, 0)

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
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['resnet', 'densenet'],
        default='resnet')
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['momentum', 'adam', 'l4'],
        default='momentum')

    return parser


# def build_train_step(loss, global_step, config):
#     assert config.optimizer in ['momentum', 'adam', 'l4']
#
#     if config.optimizer == 'momentum':
#         optimizer = tf.train.MomentumOptimizer(config.learning_rate, 0.9)
#     elif config.optimizer == 'adam':
#         optimizer = tf.train.AdamOptimizer(config.learning_rate)
#     elif config.optimizer == 'l4':
#         optimizer = L4.L4Adam(fraction=0.15)
#
#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(update_ops):
#         if config.grad_clip_norm is not None:
#             params = tf.trainable_variables()
#             gradients = tf.gradients(loss, params)
#             clipped_gradients, _ = tf.clip_by_global_norm(gradients, config.grad_clip_norm)
#             return optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
#         else:
#             return optimizer.minimize(loss, global_step=global_step)

# TODO: refactor this
def build_train_step(loss, global_step, config):
    assert config.optimizer in ['momentum', 'adam', 'l4']

    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)

    if config.grad_clip_norm is not None:
        gradients, _ = tf.clip_by_global_norm(gradients, config.grad_clip_norm)

    if config.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(config.learning_rate, 0.9)
    elif config.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
    elif config.optimizer == 'l4':
        optimizer = L4.L4Adam(fraction=0.15)
    else:
        raise AssertionError('invalid optimizer type: {}'.format(config.optimizer))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        return optimizer.apply_gradients(zip(gradients, params), global_step=global_step)


def build_metrics(total_loss, class_loss, regr_loss, regularization_loss, image, labels, logits, learning_rate):
    def build_iou(labels, logits, name='build_iou'):
        with tf.name_scope(name):
            # decode both using ground true classification
            labels_decoded = utils.boxes_decode(labels['classifications'], labels['regressions_postprocessed'])
            logits_decoded = utils.boxes_decode(labels['classifications'], logits['regressions_postprocessed'])
            return utils.iou(labels_decoded['boxes'], logits_decoded['boxes'])

    image = image * dataset.STD + dataset.MEAN
    metrics = {}
    update_metrics = {}

    # TODO: refactor
    metrics['class_iou'], update_metrics['class_iou'] = tf.metrics.mean_iou(
        labels=labels['detection_trainable']['classifications'],
        predictions=tf.to_int32(tf.nn.sigmoid(logits['detection_trainable']['classifications']) > 0.5),
        num_classes=2)
    metrics['class_pr_auc'], update_metrics['class_pr_auc'] = tf.metrics.auc(
        labels=labels['detection_trainable']['classifications'],
        predictions=tf.nn.sigmoid(logits['detection_trainable']['classifications']),
        num_thresholds=10,
        curve='PR')
    metrics['regr_iou'], update_metrics['regr_iou'] = tf.metrics.mean(
        build_iou(labels['detection_trainable'], logits['detection_trainable']))
    metrics['total_loss'], update_metrics['total_loss'] = tf.metrics.mean(total_loss)
    metrics['class_loss'], update_metrics['class_loss'] = tf.metrics.mean(class_loss)
    metrics['regr_loss'], update_metrics['regr_loss'] = tf.metrics.mean(regr_loss)
    metrics['regularization_loss'], update_metrics['regularization_loss'] = tf.metrics.mean(regularization_loss)

    running_summary = tf.summary.merge([
        tf.summary.scalar('class_iou', metrics['class_iou']),
        tf.summary.scalar('class_pr_auc', metrics['class_pr_auc']),
        tf.summary.scalar('regr_iou', metrics['regr_iou']),
        tf.summary.scalar('total_loss', metrics['total_loss']),
        tf.summary.scalar('class_loss', metrics['class_loss']),
        tf.summary.scalar('regr_loss', metrics['regr_loss']),
        tf.summary.scalar('regularization_loss', metrics['regularization_loss']),
        tf.summary.scalar('learning_rate', learning_rate),
    ])

    image_summary = []
    # TODO: better scope names
    for scope, classifications, regressions in (
            ('true',
             labels['detection']['classifications'],
             labels['detection']['regressions_postprocessed']),
            ('pred',
             utils.dict_map(tf.nn.sigmoid, logits['detection']['classifications']),
             logits['detection']['regressions_postprocessed'])
    ):
        for i in range(image.shape[0]):
            with tf.name_scope('{}/{}'.format(scope, i)):
                image_with_boxes = draw_bounding_boxes(
                    image[i],
                    utils.dict_map(lambda x: x[i], classifications),
                    utils.dict_map(lambda x: x[i], regressions))
                image_summary.append(tf.summary.image('regression', tf.expand_dims(image_with_boxes, 0)))

                image_with_classmap = draw_classmap(
                    image[i], utils.dict_map(lambda x: x[i], classifications))
                image_summary.append(tf.summary.image('classification', tf.expand_dims(image_with_classmap, 0)))

    image_summary = tf.summary.merge(image_summary)

    return metrics, update_metrics, running_summary, image_summary


def main():
    args = build_parser().parse_args()
    utils.log_args(args)

    levels = build_levels()
    training = tf.placeholder(tf.bool, [], name='training')
    global_step = tf.get_variable('global_step', initializer=0, trainable=False)

    ds = dataset.build_dataset(
        spec=args.dataset,
        levels=levels,
        scale=args.scale,
        shuffle=1024,
        augment=True)

    iter = ds['dataset'].prefetch(1).make_initializable_iterator()
    input = iter.get_next()
    input = {
        **input,
        'image': preprocess_image(input['image'])
    }

    net = retinanet.RetinaNet(
        levels=levels,
        num_classes=ds['num_classes'],
        dropout_rate=args.dropout,
        backbone=args.backbone)
    logits = {'detection': net(input['image'], training)}
    image_size = tf.shape(input['image'])[1:3]
    input = utils.apply_trainable_masks(input, input['trainable_masks'], image_size=image_size, levels=levels)
    logits = utils.apply_trainable_masks(logits, input['trainable_masks'], image_size=image_size, levels=levels)

    class_loss, regr_loss = losses.loss(labels=input['detection_trainable'], logits=logits['detection_trainable'])
    regularization_loss = tf.losses.get_regularization_loss()

    total_loss = class_loss + regr_loss + regularization_loss
    train_step = build_train_step(total_loss, global_step=global_step, config=args)

    metrics, update_metrics, running_summary, image_summary = build_metrics(
        total_loss,
        class_loss,
        regr_loss,
        regularization_loss,
        image=input['image'],
        labels=input,
        logits=logits,
        learning_rate=args.learning_rate)

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
                        m, run_summ, img_summ = sess.run(
                            [metrics, running_summary, image_summary], {training: True})

                        print()
                        print_summary(m, step)
                        train_writer.add_summary(run_summ, step)
                        train_writer.add_summary(img_summ, step)
                        saver.save(sess, os.path.join(args.experiment, 'model.ckpt'), write_meta_graph=False)
                        sess.run(locals_init)
                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    main()
