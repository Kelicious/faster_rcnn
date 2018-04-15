import argparse
import numpy as np
import os

from data.voc_data_helpers import extract_img_data, KITTI_CLASS_MAPPING, VOC_CLASS_MAPPING


def voc_ap(rec, prec, use_07_metric=False):
    ap = 0.
    if use_07_metric:
        # 11 point metric
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(voc_path, det_file, imageset_path, cls_name, ovthresh=0.5):
    with open(imageset_path, 'r') as f:
        imagenames = [line.strip() for line in f.readlines()]

    gt_boxes_by_imagename = {}
    for i, imagename in enumerate(imagenames):
        if i % 100 == 0:
            print('Reading annotation for image {}/{}'.format(i, len(imagenames)))
        img = extract_img_data(voc_path, imagename)
        gt_boxes_by_imagename[imagename] = img.gt_boxes

    # extract gt objects for this class
    class_recs = {}
    npos = 0

    for imagename in imagenames:
        R = [box for box in gt_boxes_by_imagename[imagename] if box.obj_cls == cls_name]
        bbox = np.array([box.corners for box in R])
        difficult = np.array([box.difficult for box in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # read dets
    with open(det_file, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric=True)

    return rec, prec, ap


def get_voc_results_filename(dets_path, cls_name):
    det_file = os.path.join(dets_path, 'comp3_det_test_{}.txt'.format(cls_name))

    return det_file


def eval_all(dets_path, voc_path, class_mapping, img_set='val'):
    aps = []
    for cls_name, cls_num in sorted(class_mapping.items()):
        print(cls_name)
        if cls_name == 'bg':
            continue

        det_file = get_voc_results_filename(dets_path, cls_name)
        imageset_file = os.path.join(voc_path, 'ImageSets', 'Main', img_set + '.txt')
        rec, prec, ap = voc_eval(voc_path, det_file, imageset_file, cls_name, ovthresh=0.5)
        aps.append(ap)
        print('AP for {} = {:.4f}'.format(cls_name, ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Placeholder description')
    DEFAULT_VOC_PATH = '/Users/ke/mlnd/vocstuff/VOCdevkit/VOC2007'
    parser.add_argument('--voc_path', dest='voc_path',
                        help='Base path of the VOC test dataset',
                        default=DEFAULT_VOC_PATH)
    parser.add_argument('--dets_path', dest='dets_path',
                        help='Directory containing detection outputs',
                        default='./tmpout')
    parser.add_argument('--kitti', dest='kitti',
                        help='Using KITTI dataset or not, otherwise defaults to Pascal VOC classes',
                        action='store_true')
    parser.add_argument('--img_set', dest='img_set', choices=('val', 'test'),
                        help='which image set to use, val or test',
                        default='val')

    args = parser.parse_args()

    class_mapping = KITTI_CLASS_MAPPING if args.kitti else VOC_CLASS_MAPPING
    eval_all(dets_path=args.dets_path, voc_path=args.voc_path, class_mapping=class_mapping, img_set=args.img_set)
