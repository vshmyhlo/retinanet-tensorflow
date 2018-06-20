import argparse
import tensorflow as tf
import dataset
import utils
from level import build_levels
import losses
from train import build_train_step, preprocess_image, build_metrics, build_summary
import retinanet


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, nargs='+', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--scale', type=int, default=600)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--grad-clip-norm', type=float)
    parser.add_argument('--ohem', type=int)
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


def train_input_fn(spec, levels, scale):
    def split(input):
        features = {
            'image': input['image'],
        }

        labels = {
            'detection': input['detection'],
            'trainable_masks': input['trainable_masks']
        }

        return features, labels

    ds = dataset.build_dataset(
        spec=spec,
        levels=levels,
        scale=scale,
        shuffle=1024,
        augment=True)

    # return ds['dataset'].map(split).prefetch(1) # FIXME: not for tf 1.4
    return ds['dataset'].map(split).prefetch(1).make_one_shot_iterator().get_next()


def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True
        global_step = tf.train.get_global_step()

        features = {
            **features,
            'image': preprocess_image(features['image'])
        }

        net = retinanet.RetinaNet(
            levels=levels,
            # num_classes=config.num_classes, # FIXME:
            num_classes=20,
            activation=tf.nn.elu,
            dropout_rate=params.dropout,
            backbone=params.backbone)
        logits = {'detection': net(features['image'], training)}
        image_size = tf.shape(features['image'])[1:3]
        labels = utils.apply_trainable_masks(
            labels, labels['trainable_masks'], image_size=image_size, levels=levels)
        logits = utils.apply_trainable_masks(logits, labels['trainable_masks'], image_size=image_size, levels=levels)

        class_loss, regr_loss = losses.loss(
            labels=labels['detection_trainable'], logits=logits['detection_trainable'], top_k=params.ohem)
        regularization_loss = tf.losses.get_regularization_loss()

        loss = class_loss + regr_loss + regularization_loss
        train_step = build_train_step(loss, global_step=global_step, config=params)

        metrics, update_metrics = build_metrics(
            loss,
            class_loss,
            regr_loss,
            regularization_loss,
            labels=labels,
            logits=logits)

        running_summary, image_summary = build_summary(
            metrics,
            image=features['image'],
            labels=labels,
            logits=logits,
            learning_rate=params.learning_rate,
            class_names=[
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ])  # FIXME:

        def merge_metrics(metrics, update_metrics):
            if isinstance(metrics, dict):
                return {k: merge_metrics(metrics[k], update_metrics[k]) for k in metrics}
            else:
                return metrics, update_metrics

        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_step, eval_metric_ops=merge_metrics(metrics, update_metrics))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    params = build_parser().parse_args()
    utils.log_args(params)
    levels = build_levels()

    config = tf.estimator.RunConfig(
        save_summary_steps=params.log_interval)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=config,
        model_dir=params.experiment)

    for _ in range(params.epochs):
        classifier.train(lambda: train_input_fn(params.dataset, levels, params.scale))
