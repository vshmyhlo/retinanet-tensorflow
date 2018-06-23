import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from data_loaders.base import Base


class Pascal(Base):
    def __init__(self, path, subset):  # FIXME:
        self._path = path
        self._subset = subset
        self._class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    @property
    def class_names(self):
        return self._class_names

    @property
    def num_classes(self):
        return len(self._class_names)

    def __iter__(self):
        with open(os.path.join(self._path, 'ImageSets', 'Main', self._subset + '.txt')) as f:
            lines = f.readlines()
            image_names = [line.strip().split()[0] for line in lines]

        for image_name in image_names:
            image_file = os.path.join(self._path, 'JPEGImages', image_name + '.jpg')
            tree = ET.parse(os.path.join(self._path, 'Annotations', image_name + '.xml'))

            boxes = []
            class_ids = []
            for obj in tree.getroot().iter('object'):
                t = float(obj.find('bndbox/ymin').text)
                l = float(obj.find('bndbox/xmin').text)
                b = float(obj.find('bndbox/ymax').text)
                r = float(obj.find('bndbox/xmax').text)

                boxes.append([t, l, b, r])
                class_ids.append(self._class_names.index(obj.find('name').text))

            boxes = np.array(boxes).reshape((-1, 4))
            class_ids = np.array(class_ids).reshape(-1)

            yield {
                'image_file': image_file.encode('utf-8'),
                'class_ids': class_ids,
                'boxes': boxes
            }


if __name__ == '__main__':
    dl = Pascal(os.path.expanduser('~/Datasets/pascal/VOCdevkit/VOC2012'), 'trainval')

    print(dl.class_names)

    for x in tqdm(dl):
        pass

    print(x)
