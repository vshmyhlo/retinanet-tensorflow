import numpy as np
import cv_utils


def test_iou():
    iou = cv_utils.iou(
        np.array([
            [0.1, 0.1, 0.2, 0.2],
            [100, 100, 200, 200],
            [0.1, 0.1, 0.2, 0.2],
            [1., 1., 1., 1.],
        ]),
        np.array([
            [0.1, 0.1, 0.3, 0.3],
            [100, 100, 300, 300],
            [100, 100, 300, 300],
            [0., 0., 0., 0.],
        ]))

    assert np.allclose(iou, [0.25, 0.25, 0, 0])
