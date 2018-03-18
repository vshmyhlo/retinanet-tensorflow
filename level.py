from itertools import product
import numpy as np


def compute_box_size(base_size, aspect_ratio, scale_ratio):
    aspect_ratio = np.array(aspect_ratio)
    size = np.sqrt(
        base_size**2 / aspect_ratio.prod()) * aspect_ratio * scale_ratio
    return size


class Level(object):
    def __init__(self, number, anchor_size, anchor_aspect_ratios,
                 anchor_scale_ratios):
        self._number = number
        self._anchor_size = anchor_size
        self._anchor_aspect_ratios = anchor_aspect_ratios
        self._anchor_scale_ratios = anchor_scale_ratios

    @property
    def number(self):
        return self._number

    @property
    def anchor_boxes(self):
        return np.stack([
            compute_box_size(self._anchor_size, aspect_ratio, scale_ratio)
            for aspect_ratio, scale_ratio in product(
                self._anchor_aspect_ratios, self._anchor_scale_ratios)
        ], 0)

    def __repr__(self):
        return 'Level(number={}, anchor_size={}, anchor_aspect_ratios={}, anchor_scale_ratios={})'.format(
            self.number, self._anchor_size, self._anchor_aspect_ratios,
            self._anchor_scale_ratios)


def make_levels():
    anchor_aspect_ratios = [(1, 2), (1, 1), (2, 1)]
    anchor_scale_ratios = [2**0, 2**(1 / 3), 2**(2 / 3)]

    return [
        Level(3, 32, anchor_aspect_ratios, anchor_scale_ratios),
        Level(4, 64, anchor_aspect_ratios, anchor_scale_ratios),
        Level(5, 128, anchor_aspect_ratios, anchor_scale_ratios),
        Level(6, 256, anchor_aspect_ratios, anchor_scale_ratios),
        Level(7, 512, anchor_aspect_ratios, anchor_scale_ratios),
    ]
