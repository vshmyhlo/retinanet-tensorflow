import dataset
import numpy as np


def test_compute_box_size():
    box_size = dataset.compute_box_size(32, (1, 2), 1)
    assert len(box_size) == 2
    assert np.isclose(box_size[0] * box_size[1], 32**2)
    assert box_size[1] / box_size[0] == 2
