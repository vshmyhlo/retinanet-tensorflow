from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loaders.inferred import Inferred
import utils
import cv2

if __name__ == '__main__':
    # dl = Inferred('pascal', [os.path.expanduser('~/Datasets/pascal/VOCdevkit/VOC2012'), 'trainval'])
    # dl = Inferred('coco', [os.path.expanduser('~/Datasets/coco/instances_train2017.json'),
    #                        os.path.expanduser('~/Datasets/coco/images')])
    dl = Inferred('shapes', ['./tmp', 10, 600])

    for x in tqdm(dl):
        image = cv2.imread(x['image_file'].decode('utf-8'))
        image = utils.draw_bounding_boxes(image, x['boxes'] / [600, 600, 600, 600], x['class_ids'], dl.class_names)
        plt.imshow(image)
        plt.show()

        break
