import argparse
import numpy as np
import pandas as pd
from args_util import base_paths_to_imgs, resize_dims_from_str
from util import resize_imgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Placeholder description')
    DEFAULT_VOC_PATH = '/Users/ke/mlnd/vocstuff/VOCdevkit/VOC2007'
    parser.add_argument('--voc_paths', dest='voc_paths',
                        help='Base paths of the VOC dataset(s), comma separated if multiple',
                        default=DEFAULT_VOC_PATH)
    parser.add_argument('--resize_dims', dest='resize_dims',
                        help='resize parameters, e.g. 600,1000 if resizing to a min size of 600 pixels and max 1000 pixels',
                        default="600,1000")
    parser.add_argument('--obj_cls', dest='obj_cls',
                        help='Specific object class to get stats for, all classes if not supplied',
                        default=None)
    parser.add_argument('--img_set', dest='img_set', choices=('train', 'trainval', 'val', 'test'),
                        help='which image set to use, choose one of train, trainval, val, or test',
                        default='trainval')

    args = parser.parse_args()

    percentiles = np.arange(0.0, 1.0, 0.05)

    train_imgs = base_paths_to_imgs(args.voc_paths, img_set=args.img_set, do_flip=False)
    resize_min, resize_max = resize_dims_from_str(args.resize_dims)
    processed_imgs, resized_ratios = resize_imgs(train_imgs, min_size=resize_min, max_size=resize_max)
    gt_boxes = [gt_box for img in processed_imgs for gt_box in img.gt_boxes]

    if args.obj_cls is not None:
        gt_boxes = [gt_box for gt_box in gt_boxes if gt_box.obj_cls == args.obj_cls]

    print("{} objects in {} images".format(len(gt_boxes), len(train_imgs)))
    heights = np.array([gt_box.height for gt_box in gt_boxes])
    height_series = pd.Series(heights)
    print("Heights statistics:")
    print(height_series.describe(percentiles=percentiles))

    widths = np.array([gt_box.width for gt_box in gt_boxes])
    width_series = pd.Series(widths)
    print("Widths statistics:")
    print(width_series.describe(percentiles=percentiles))

    areas = heights * widths
    area_series = pd.Series(areas)
    print("Areas statistics:")
    print(area_series.describe(percentiles=percentiles))
