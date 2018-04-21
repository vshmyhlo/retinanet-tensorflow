from level import Level, compute_box_size
import numpy as np


def test_compute_box_size():
    box_size = compute_box_size(32, (1, 2), 1)
    assert len(box_size) == 2
    assert np.isclose(box_size.prod(), 32**2)
    assert box_size[1] / box_size[0] == 2


def test_level_anchor_boxes():
    level = Level(32, [(1, 4)], [2**0, 2**1])
    assert np.array_equal(level.anchor_sizes, [[16, 64], [32, 128]])
