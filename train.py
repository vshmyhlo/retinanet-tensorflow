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

def preprocess_image(image):
    return (image - dataset.MEAN) / dataset.STD


def print_summary(metrics, step):
    print(
        'step: {}, total_loss: {:.4f}, class_loss: {:.4f}, regr_loss: {:.4f}, regularization_loss: {:.4f}'.format(
            step, metrics['total_loss'], metrics['class_loss'], metrics['regr_loss'], metrics['regularization_loss']))


def draw_classmap(image, classifications):
    for pn in classifications:
        classification = classifications[pn]

        image_size = tf.shape(image)[:2]
        classification = utils.classmap_decode(classification)
        classification = tf.not_equal(classification, -1)
        classification = tf.to_float(classification)
        classification = tf.reduce_sum(classification, -1)
        classification = tf.expand_dims(classification, -1)
        classification = tf.image.resize_images(
            classification, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

        image += classification

    return image


def draw_bounding_boxes(image, classifications, regressions, max_output_size=1000):
    assert classifications.keys() == regressions.keys()

    final_boxes = []
    final_scores = []

    for pn in classifications:
        non_background_mask = tf.not_equal(utils.classmap_decode(classifications[pn]), -1)
        boxes = tf.boolean_mask(regressions[pn], non_background_mask)
        scores = tf.reduce_max(classifications[pn], -1)
        scores = tf.boolean_mask(scores, non_background_mask)

        final_boxes.append(boxes)
        final_scores.append(scores)

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


# def class_distribution(tensors):
#     # TODO: do not average over batch
#     return tf.stack([
#         tf.reduce_mean(tf.to_float(tf.argmax(tensors[k], -1)), [0, 1, 2])
#         for k in tensors
#     ])


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

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
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

        return optimizer.apply_gradients(zip(gradients, params), global_step=global_step)


def build_metrics(total_loss, class_loss, regr_loss, regularization_loss, image, true, pred, trainable_masks, levels,
                  learning_rate):
    # TODO: refactor
    def build_iou(labels, logits, classifications_true):  # TODO: trainable_mask
        classifications_true = utils.merge_outputs(classifications_true, trainable_masks)
        labels = utils.merge_outputs(labels, trainable_masks)
        logits = utils.merge_outputs(logits, trainable_masks)

        non_background_mask = tf.not_equal(utils.classmap_decode(classifications_true), -1)

        labels = tf.boolean_mask(labels, non_background_mask)
        logits = tf.boolean_mask(logits, non_background_mask)

        return utils.iou(labels, logits)

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

    regr_iou = build_iou(regressions_true, regressions_pred, classifications_true)

    metrics = {}
    update_metrics = {}

    metrics['regr_iou'], update_metrics['regr_iou'] = tf.metrics.mean(regr_iou)
    metrics['total_loss'], update_metrics['total_loss'] = tf.metrics.mean(total_loss)
    metrics['class_loss'], update_metrics['class_loss'] = tf.metrics.mean(class_loss)
    metrics['regr_loss'], update_metrics['regr_loss'] = tf.metrics.mean(regr_loss)
    metrics['regularization_loss'], update_metrics['regularization_loss'] = tf.metrics.mean(regularization_loss)
    metrics['logits_grad_fg'], update_metrics['logits_grad_fg'] = tf.metrics.mean(tf.get_collection('logits_grad_fg'))
    metrics['logits_grad_bg'], update_metrics['logits_grad_bg'] = tf.metrics.mean(tf.get_collection('logits_grad_bg'))
    # running_true_class_dist, update_true_class_dist = tf.metrics.mean_tensor(
    #     class_distribution(classifications_true))
    # running_pred_class_dist, update_pred_class_dist = tf.metrics.mean_tensor(
    #     class_distribution(classifications_pred))

    running_summary = tf.summary.merge([
        tf.summary.scalar('regr_iou', metrics['regr_iou']),
        tf.summary.scalar('total_loss', metrics['total_loss']),
        tf.summary.scalar('class_loss', metrics['class_loss']),
        tf.summary.scalar('regr_loss', metrics['regr_loss']),
        tf.summary.scalar('regularization_loss', metrics['regularization_loss']),
        tf.summary.scalar('learning_rate', learning_rate),
        tf.summary.scalar('logits_grad_fg', metrics['logits_grad_fg']),
        tf.summary.scalar('logits_grad_bg', metrics['logits_grad_bg'])
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
                    image[i],
                    {pn: classifications[pn][i] for pn in classifications},
                    {pn: regressions[pn][i] for pn in regressions})
                image_summary.append(tf.summary.image('regression', tf.expand_dims(image_with_boxes, 0)))

                image_with_classmap = draw_classmap(image, {pn: classifications[pn][i] for pn in classifications})
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
        augment=True)

    iter = ds['dataset'].shuffle(32).prefetch(1).make_initializable_iterator()
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
    classifications_pred, regressions_pred = net(input['image'], training)
    assert input['classifications'].keys() == input['regressions'].keys() == levels.keys()
    assert classifications_pred.keys() == regressions_pred.keys() == levels.keys()

    class_loss, regr_loss = losses.loss(
        (input['classifications'], input['regressions']),
        (classifications_pred, regressions_pred),
        trainable_masks=input['trainable_masks'])
    regularization_loss = tf.losses.get_regularization_loss()

    total_loss = class_loss + regr_loss + regularization_loss
    train_step = build_train_step(total_loss, global_step=global_step, config=args)

    metrics, update_metrics, running_summary, image_summary = build_metrics(
        total_loss,
        class_loss,
        regr_loss,
        regularization_loss,
        image=input['image'],
        true=(input['classifications'], input['regressions']),
        pred=(classifications_pred, regressions_pred),
        trainable_masks=input['trainable_masks'],
        levels=levels,
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
