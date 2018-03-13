import numpy as np
import pycocotools.coco as pycoco


class COCO(object):
    class Image(object):
        def __init__(self, img):
            self.id = img['id']
            self.filename = img['file_name']
            self.size = np.array(
                [img['height'], img['width']], dtype=np.float32)

    class Annotation(object):
        def __init__(self, ann, category_ids):
            [left, top, width, height] = ann['bbox']
            self.bbox = np.array(
                [top + height / 2, left + width / 2, height, width],
                dtype=np.float32)
            self.category_id = category_ids.index(ann['category_id'])

    def __init__(self, ann_path, dataset_path, download):
        self.coco = pycoco.COCO(ann_path)
        self.category_ids = ['BG'] + sorted(self.coco.getCatIds())
        self.num_classes = len(self.category_ids)

        if download:
            self.coco.download(tarDir=dataset_path)

    def get_img_ids(self):
        return self.coco.getImgIds()

    def load_imgs(self, ids):
        return [self.Image(img) for img in self.coco.loadImgs(ids=ids)]

    def get_ann_ids(self, img_ids):
        return self.coco.getAnnIds(imgIds=img_ids)

    def load_anns(self, ids):
        return [
            self.Annotation(ann, self.category_ids)
            for ann in self.coco.loadAnns(ids=ids)
        ]
