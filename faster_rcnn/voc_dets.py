import argparse
import numpy as np
import os
import timeit

import resnet
import vgg

from data.voc_data_helpers import KITTI_CLASS_MAPPING, VOC_CLASS_MAPPING
from det_util import DetTrainingManager
from shared_constants import BBREG_MULTIPLIERS
from det_util import nms
from args_util import base_paths_to_imgs, resize_dims_from_str, anchor_scales_from_str
from util import resize_imgs, transform, get_anchors


DEFAULT_DET_THRESHOLD = 0.0


def get_dets(training_manager, detector, image, resize_ratio, num_rois=64, stride=16,
             det_threshold=DEFAULT_DET_THRESHOLD):
    conv_out, rois = training_manager.get_det_inputs(image)
    class_mapping = training_manager.class_mapping
    rev_class_mapping = dict((v,k) for k,v in class_mapping.items())

    num_boxes = rois.shape[0]
    print("num rois: {}".format(num_boxes))
    # remove this later
    # num_boxes = 64

    num_batches = num_boxes // num_rois + 1 if num_boxes % num_rois != 0 else num_boxes // num_rois
    bg_idx = class_mapping['bg']

    bboxes_by_cls = {}
    probs_by_cls = {}

    for batch_num in range(num_batches):
        start_idx = batch_num * num_rois
        end_idx = start_idx + num_rois
        batch_rois = rois[start_idx:end_idx, :]

        if batch_num == num_batches - 1:
            # add repeat ROIs because the network expects exactly 64 in a batch
            num_to_add = num_rois - batch_rois.shape[0]
            extra_rois = np.tile(batch_rois[0], (num_to_add, 1))
            batch_rois = np.concatenate([batch_rois, extra_rois])

        batch_rois = np.expand_dims(batch_rois, axis=0)
        out_cls, out_reg = detector.predict([conv_out, batch_rois])

        for roi_idx in range(num_rois):
            cls_idx = np.argmax(out_cls[0, roi_idx])
            confidence = out_cls[0, roi_idx, cls_idx]
            # print(batch_rois[0, roi_idx])
            # print(cls_idx)
            # print(confidence)
            if cls_idx == bg_idx or confidence < det_threshold:
                continue

            cls_name = rev_class_mapping[cls_idx]
            if cls_name not in bboxes_by_cls:
                bboxes_by_cls[cls_name] = []
                probs_by_cls[cls_name] = []

            x1, y1, x2, y2 = batch_rois[0, roi_idx]
            tx, ty, tw, th = out_reg[0, roi_idx, cls_idx * 4 : (cls_idx+1) * 4] / BBREG_MULTIPLIERS

            px1, py1, px2, py2 = transform([x1, y1, x2, y2], [tx, ty, tw, th])
            bboxes_by_cls[cls_name].append([stride * px1, stride * py1, stride * px2, stride * py2])
            probs_by_cls[cls_name].append(out_cls[0, roi_idx, cls_idx])

    dets = []
    for cls_name in bboxes_by_cls:
        bboxes = np.array(bboxes_by_cls[cls_name])
        probs = np.array(probs_by_cls[cls_name])
        new_boxes, new_probs = nms(bboxes, probs, overlap_thresh=0.5, max_boxes=2000)
        for box_idx in range(new_boxes.shape[0]):
            x1, y1, x2, y2 = new_boxes[box_idx,:]
            real_x1 = int(round(x1 / resize_ratio))
            real_y1 = int(round(y1 / resize_ratio))
            real_x2 = int(round(x2 / resize_ratio))
            real_y2 = int(round(y2 / resize_ratio))

            det_obj = {'bbox': np.array([real_x1, real_y1, real_x2, real_y2]),
                       'cls_name': cls_name, 'prob': new_probs[box_idx]}
            dets.append(det_obj)

    return dets


def get_dets_by_cls(training_manager, detector, resized_ratios, images, stride=16, det_threshold=DEFAULT_DET_THRESHOLD):
    dets_by_cls = {}
    for image, resized_ratio in zip(images, resized_ratios):

        start_time = timeit.default_timer()
        dets = get_dets(training_manager, detector, image, resized_ratio, stride=stride,
                        det_threshold=det_threshold)
        print(dets)
        for det in dets:
            if det['cls_name'] not in dets_by_cls:
                dets_by_cls[det['cls_name']] = {}

            cls_dets = dets_by_cls[det['cls_name']]
            if image.name not in cls_dets:
                cls_dets[image.name] = []

            cls_dets[image.name].append(det)
        end_time = timeit.default_timer()
        print("image {} ran in {} seconds".format(image.name, end_time - start_time))

    return dets_by_cls


def write_dets(dets, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for cls_name in dets:
        cls_dets = dets[cls_name]
        basename = "comp3_det_test_{}.txt".format(cls_name)

        file_path = os.path.join(out_dir, basename)

        with open(file_path, 'w') as outfile:
            for image_name in cls_dets:
                image_dets = cls_dets[image_name]
                for det in image_dets:
                    x1, y1, x2, y2 = det['bbox'] + 1
                    output_line = "{} {} {} {} {} {}\n".format(
                        image_name,det['prob'], x1, y1, x2, y2)
                    outfile.write(output_line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Placeholder description')
    DEFAULT_VOC_PATH = '/Users/ke/Downloads/VOCdevkit/VOC2007'
    parser.add_argument('step3_model_path', metavar='step3_model_path', type=str,
                        help='Path to the h5 file holding the rpn model with weights trained in step 3.')
    parser.add_argument('step4_model_path', metavar='step4_model_path', type=str,
                        help='Path to the h5 file holding the detector model from step 4. Must be compatible with the rpn.')
    parser.add_argument('--voc_path', dest='voc_path',
                        help='Base path of the VOC test set',
                        default=DEFAULT_VOC_PATH)
    parser.add_argument('--kitti', dest='kitti',
                        help='Using KITTI dataset or not, otherwise defaults to Pascal VOC classes',
                        action='store_true')
    parser.add_argument('--img_set', dest='img_set', choices=('val', 'test'),
                        help='which image set to use, val or test',
                        default="val")
    parser.add_argument('--resize_dims', dest='resize_dims',
                        help='resize parameters, e.g. 600,1000 if resizing to a min size of 600 pixels and max 1000 pixels',
                        default="600,1000")
    parser.add_argument('--anchor_scales', dest='anchor_scales',
                        help='anchor scales in pixels, e.g. 128,256,512 if following the original paper',
                        default="128,256,512")
    parser.add_argument('--network', dest='network', choices=('vgg16', 'resnet50', 'resnet101'),
                        help='underlying network architecture, choose from vgg16, resnet50 or resnet101',
                        default="vgg16")
    parser.add_argument('--out_dir', dest='out_dir',
                        help='Location to write the output files',
                        default='.')
    parser.add_argument('--det_threshold', dest='det_threshold',
                        help='Minimum confidence level (from 0 to 1) needed to output a detection',
                        default=DEFAULT_DET_THRESHOLD)

    args = parser.parse_args()
    det_threshold = float(args.det_threshold)

    test_imgs = base_paths_to_imgs(args.voc_path, img_set=args.img_set, do_flip=False)
    anchor_scales = anchor_scales_from_str(args.anchor_scales)
    anchors = get_anchors(anchor_scales)
    anchors_per_loc = len(anchors)
    print("num test_imgs: ", len(test_imgs))
    class_mapping = KITTI_CLASS_MAPPING if args.kitti else VOC_CLASS_MAPPING
    num_classes = len(class_mapping)

    if args.network == 'vgg16':
        # don't need to worry about freezing/regularizing rpn because we're not training it
        model_rpn = vgg.rpn_from_h5(args.step3_model_path, anchors_per_loc=anchors_per_loc)
        model_det = vgg.det_from_h5(args.step4_model_path, num_classes=num_classes)
        stride = vgg.STRIDE
    else:
        model_rpn = resnet.rpn_from_h5(args.step3_model_path, anchors_per_loc=anchors_per_loc)
        model_det = resnet.det_from_h5(args.step4_model_path, num_classes=num_classes)
        stride = resnet.STRIDE
    training_manager = DetTrainingManager(rpn_model=model_rpn, class_mapping=class_mapping,
                                          preprocess_func=resnet.preprocess, anchor_dims=anchors)
    resize_min, resize_max = resize_dims_from_str(args.resize_dims)
    processed_imgs, resized_ratios = resize_imgs(test_imgs, min_size=resize_min, max_size=resize_max)

    dets = get_dets_by_cls(training_manager, model_det, resized_ratios, processed_imgs, stride=stride,
                           det_threshold=det_threshold)
    print(dets)
    write_dets(dets, args.out_dir)
