import numpy as np
import pycocotools.coco as pycoco
from data_loaders.base import Base
import os
from tqdm import tqdm


class COCO(Base):
    def __init__(self, ann_path, dataset_path):
        self._coco = pycoco.COCO(ann_path)
        self._dataset_path = dataset_path
        self._category_ids = sorted(self._coco.getCatIds())

    @property
    def class_names(self):
        categories = self._coco.loadCats(ids=self._category_ids)
        return [c['name'] for c in categories]

    @property
    def num_classes(self):
        return len(self._category_ids)

    def download(self):
        self._coco.download(tarDir=self._dataset_path)

    def __iter__(self):
        image_ids = self._coco.getImgIds()
        images = self._coco.loadImgs(ids=image_ids)
        for image in images:
            image_file = os.path.join(self._dataset_path, image['file_name'])
            annotation_ids = self._coco.getAnnIds(imgIds=image['id'], iscrowd=False)
            annotations = self._coco.loadAnns(ids=annotation_ids)

            boxes = []
            class_ids = []

            for a in annotations:
                [left, top, width, height] = a['bbox']

                # some boxes have no width / height
                if height < 1 or width < 1:
                    continue

                assert height > 0, 'height {} <= 0'.format(height)  # FIXME:
                assert width > 0, 'width {} <= 0'.format(width)  # FIXME:
                boxes.append([top, left, top + height, left + width])
                class_ids.append(self._category_ids.index(a['category_id']))

            boxes = np.array(boxes).reshape((-1, 4))  # TODO: normalize boxes
            class_ids = np.array(class_ids).reshape(-1)

            # ignore samples without boxes
            if len(annotations) > 0:
                yield {
                    'image_file': image_file.encode('utf-8'),
                    'class_ids': class_ids,
                    'boxes': boxes
                }


if __name__ == '__main__':
    dl = COCO(
        os.path.expanduser('~/Datasets/coco/instances_train2017.json'),
        os.path.expanduser('~/Datasets/coco/images'))

    print(dl.class_names)

    for x in tqdm(dl):
        pass

    print(x)
