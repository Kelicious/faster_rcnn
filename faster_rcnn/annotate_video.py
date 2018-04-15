import argparse
import cv2
import pathlib
import os

from data.voc_data_helpers import KITTI_CLASS_MAPPING, VOC_CLASS_MAPPING
from det_util import DetTrainingManager
from shapes import InMemoryImage
from resnet import rpn_from_h5, det_from_h5, preprocess
from args_util import resize_dims_from_str
from util import resize_imgs
from voc_dets import get_dets


def annotate_images(training_manager, detector, input_dir, out_dir, image_filenames, resize_min, resize_max):
    for image_filename in image_filenames:
        image_path = os.path.join(input_dir, image_filename)
        print("processing {}".format(image_path))
        frame = cv2.imread(image_path)
        height, width = frame.shape[0:2]
        img = InMemoryImage(data=frame, width=width, height=height)
        annotated_frame = get_annotated_frame(training_manager, detector, frame, img, resize_min, resize_max)
        out_path = os.path.join(out_dir, image_filename)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, annotated_frame)

def get_annotated_frame(training_manager, detector, frame, img, resize_min, resize_max):
    resized_imgs, resized_ratios = resize_imgs([img], min_size=resize_min, max_size=resize_max)
    dets = get_dets(training_manager=training_manager, detector=detector, image=resized_imgs[0],
                    resize_ratio=resized_ratios[0], det_threshold=0.0)

    for det in dets:
        if det['cls_name'] == 'DontCare' or det['cls_name'] == 'Misc':
            continue
        x1, y1, x2, y2 = det['bbox']
        if x1 < 0 or x2 > img.width or y1 < 0 or y2 > img.height:
            # boxes that cross image boundaries aren't necessarily bad detections, just don't want them in the video
            continue
        print(det)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), thickness=3)
        annotation = "{} {:6.2f}".format(det['cls_name'], det['prob'])
        cv2.putText(frame, annotation, (x1, y2+16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), thickness=2)

    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Placeholder description')
    parser.add_argument('step3_model_path', metavar='step3_model_path', type=str,
                        help='Path to the h5 file holding the rpn model with weights trained in step 3.')
    parser.add_argument('step4_model_path', metavar='step4_model_path', type=str,
                        help='Path to the h5 file holding the detector model from step 4. Must be compatible with the rpn.')
    parser.add_argument('input_dir', type=str,
                        help='Path to the video to annotate')
    parser.add_argument('--kitti', dest='kitti',
                        help='Using KITTI dataset or not, otherwise defaults to Pascal VOC classes',
                        action='store_true')
    parser.add_argument('--resize_dims', dest='resize_dims',
                        help='resize parameters, e.g. 600,1000 if resizing to a min size of 600 pixels and max 1000 pixels',
                        default="600,1000")
    parser.add_argument('--out_dir', dest='out_dir',
                        help='Location to write the output files',
                        default='.')

    args = parser.parse_args()
    class_mapping = KITTI_CLASS_MAPPING if args.kitti else VOC_CLASS_MAPPING
    num_classes = len(class_mapping)

    rpn = rpn_from_h5(args.step3_model_path)
    detector = det_from_h5(args.step4_model_path, num_classes=num_classes)
    training_manager = DetTrainingManager(rpn_model=rpn, class_mapping=class_mapping, preprocess_func=preprocess)
    resize_min, resize_max = resize_dims_from_str(args.resize_dims)

    # can't get opencv to extract images from video on EC2 so split into images externally
    image_filenames = sorted([filename for filename in os.listdir(args.input_dir) if filename.endswith('.png')])
    annotate_images(training_manager=training_manager,
                    detector=detector,
                    input_dir=args.input_dir,
                    out_dir=args.out_dir,
                    image_filenames=image_filenames,
                    resize_min=resize_min,
                    resize_max=resize_max)
