import os
import numpy as np
from data_loaders.coco import COCO
from data_loaders.pascal import Pascal
from tqdm import tqdm


class Inferred(object):
    def __init__(self, type, args):
        if type == 'coco':
            self._dl = COCO(*args)
        elif type == 'pascal':
            self._dl = Pascal(*args)
        else:
            raise AssertionError('unknown dataset type: {}'.format(type))

    @property
    def num_classes(self):
        return self._dl.num_classes

    def __iter__(self):
        for x in self._dl:
            assert x['boxes'].shape[0] == x['class_ids'].shape[0] != 0
            tl, br = np.split(x['boxes'], 2, -1)
            assert np.all(tl < br)

            yield x


if __name__ == '__main__':
    dl1 = Inferred('pascal', [os.path.expanduser('~/Datasets/pascal/VOCdevkit/VOC2012'), 'trainval'])
    dl2 = Inferred('coco', [os.path.expanduser('~/Datasets/coco/instances_train2017.json'),
                            os.path.expanduser('~/Datasets/coco/images')])

    for _ in tqdm(dl1):
        pass

    for _ in tqdm(dl2):
        pass
