import os
import argparse
import itertools
import tensorflow as tf
import utils
import retinanet
from levels import build_levels
import logging
import losses
import dataset
from tqdm import tqdm
from data_loaders.inferred import Inferred
import math


# TODO: THE CODE IN THIS FILE IS IN THE PROCESS IF BEING REFACTORED TO USE TO tf.Estimator

# TODO: rename bn prop to norm
# TODO: training arg to norm
# TODO: use GroupNormalization
# TODO: refactor to use Detection class
# TODO: check dropout usage
# TODO: add correct dropout everywhere (noise shape)
# TODO: rename c5_from_p4 layers to p4_to_c5
# TODO: remove redundant `strides` argument from conv
# TODO: remove tfe.Network
# TODO: check fpn relu activation usage
# TODO: check activation parameter usage
# TODO: maybe add some typing
# TODO: compute only loss on train
# TODO: use classwise nms
# TODO: test preprocessing
# TODO: optimize data loading
# TODO: check classmap decoded uses scaled logits (sigmoid)
# TODO: dict map vs dict starmap
# TODO: move label creation to graph?
# TODO: try focal cross-entropy
# TODO: try balanced cross-entropy
# TODO: exclude samples without prop IoU
# TODO: set network trainable parts
# TODO: visualize trainable_mask
# TODO: check if norm after dropout is ok
# TODO: check why sometimes ground true boxes not drawn
# TODO: add class iou metric
# TODO: add regr iou metrics
# TODO: explicitly set training=
# TODO: try other scheduling schemes (cyclical?)
# TODO: move drawing to utils

def preprocess_image(image):
    return (image - dataset.MEAN) / dataset.STD


def draw_classmap(image, classifications):
    for k in classifications:
        classification = classifications[k]
        fg_mask = utils.classmap_decode(classification).fg_mask
        fg_mask = tf.to_float(fg_mask)
        fg_mask = tf.reduce_sum(fg_mask, -1)
        fg_mask = tf.expand_dims(fg_mask, -1)
        image_size = tf.shape(image)[:2]
        fg_mask = tf.image.resize_images(
            fg_mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

        image += fg_mask

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
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--scale', type=int, default=600)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--grad-clip-norm', type=float)
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['resnet_50', 'densenet_121', 'densenet_169', 'mobilenet_v2'],
        default='resnet_50')
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['momentum', 'adam', 'rmsprop'],
        default='momentum')

    return parser


def build_train_step(loss, learning_rate, global_step, optimizer, grad_clip_norm):
    assert optimizer in ['momentum', 'adam', 'rmsprop']

    if optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9, 0.9)
    elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        raise AssertionError('invalid optimizer type {}'.format(optimizer))

    tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if grad_clip_norm is not None:
            grads_and_vars = optimizer.compute_gradients(loss)
            grads = [x[0] for x in grads_and_vars]
            vars = [x[1] for x in grads_and_vars]
            grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
            return optimizer.apply_gradients(zip(grads, vars), global_step=global_step)
        else:
            return optimizer.minimize(loss, global_step=global_step)


def build_metrics(total_loss, class_loss, regr_loss, regularization_loss, labels, logits):
    def build_iou(labels, logits, name='build_iou'):
        with tf.name_scope(name):
            # decode both using ground true classification
            labels_decoded = utils.boxes_decode(labels['classifications'], labels['regressions_postprocessed'])
            logits_decoded = utils.boxes_decode(labels['classifications'], logits['regressions_postprocessed'])
            return utils.iou(labels_decoded.boxes, logits_decoded.boxes)

    metrics = {}
    update_metrics = {}

    # TODO: refactor
    # TODO: fix iou to use class ids
    metrics['class_iou'], update_metrics['class_iou'] = tf.metrics.mean_iou(
        labels=labels['detection_trainable']['classifications'],
        predictions=tf.to_int32(tf.nn.sigmoid(logits['detection_trainable']['classifications']) > 0.5),
        num_classes=2)
    metrics['regr_iou'], update_metrics['regr_iou'] = tf.metrics.mean(
        build_iou(labels['detection_trainable'], logits['detection_trainable']))
    metrics['total_loss'], update_metrics['total_loss'] = tf.metrics.mean(total_loss)
    metrics['class_loss'], update_metrics['class_loss'] = tf.metrics.mean(class_loss)
    metrics['regr_loss'], update_metrics['regr_loss'] = tf.metrics.mean(regr_loss)
    metrics['regularization_loss'], update_metrics['regularization_loss'] = tf.metrics.mean(regularization_loss)

    return metrics, update_metrics


def build_summary(image, labels, logits, class_names):
    image = image * dataset.STD + dataset.MEAN

    # TODO: use better scope names
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
                tf.summary.image('regression', tf.expand_dims(image_with_boxes, 0))

                image_with_classmap = draw_classmap(
                    image[i], utils.dict_map(lambda x: x[i], classifications))
                tf.summary.image('classification', tf.expand_dims(image_with_classmap, 0))


def train_input_fn(params):
    levels = build_levels()

    ds = dataset.build_dataset(
        params['data_loader'],
        levels=levels,
        scale=params['scale'],
        shuffle=4096,
        augment=True)

    ds = ds.map(lambda input: {**input, 'image': preprocess_image(input['image'])})

    return ds.prefetch(1)


# TODO: use labels arg
def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        levels = build_levels()

        net = retinanet.RetinaNet(
            levels=levels,
            num_classes=params['data_loader'].num_classes,
            activation=tf.nn.elu,
            dropout_rate=params['dropout'],
            backbone=params['backbone'])
        logits = {'detection': net(features['image'], training=True)}
        input, logits = utils.process_labels_and_logits(labels=features, logits=logits, levels=levels)

        class_loss, regr_loss = losses.loss(labels=input['detection_trainable'], logits=logits['detection_trainable'])
        regularization_loss = tf.losses.get_regularization_loss()

        loss = class_loss + regr_loss + regularization_loss
        train_step = build_train_step(
            loss, params['learning_rate'], global_step=global_step, optimizer=params['optimizer'],
            grad_clip_norm=params['grad_clip_norm'])

        # TODO: build metrics
        # metrics, update_metrics = build_metrics(
        #     loss,
        #     class_loss,
        #     regr_loss,
        #     regularization_loss,
        #     labels=input,
        #     logits=logits)

        build_summary(
            input['image'],
            labels=input,
            logits=logits,
            class_names=params['data_loader'].class_names)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_step)


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()

    data_loader = Inferred(args.dataset[0], args.dataset[1:])
    params = {
        'data_loader': data_loader,
        'scale': args.scale,
        'dropout': args.dropout,
        'backbone': args.backbone,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'grad_clip_norm': args.grad_clip_norm
    }
    config = tf.estimator.RunConfig(model_dir=args.experiment, save_summary_steps=500, save_checkpoints_steps=500)

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    for epoch in range(args.epochs):
        print('epoch {}'.format(epoch))
        estimator.train(train_input_fn)


if __name__ == '__main__':
    main()
