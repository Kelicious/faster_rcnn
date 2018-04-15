import argparse

import resnet
import vgg
from data.voc_data_helpers import VOC_CLASS_MAPPING, KITTI_CLASS_MAPPING
from det_util import DetTrainingManager
from shared_constants import NUM_ROIS
from args_util import phases_from_str, optimizer_from_str, base_paths_to_imgs, resize_dims_from_str, anchor_scales_from_str
from train_util import train_detector_step4
from util import resize_imgs, get_anchors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Placeholder description')
    DEFAULT_VOC_PATH = '/Users/ke/Downloads/VOCdevkit/VOC2007'
    parser.add_argument('step3_weights_path', metavar='step3_weights_path', type=str,
                        help='Path to the h5 file holding the rpn weights from step 3. Must be compatible with the network.')
    parser.add_argument('--voc_paths', dest='voc_paths',
                        help='Base paths of the VOC dataset(s), comma separated if multiple',
                        default=DEFAULT_VOC_PATH)
    parser.add_argument('--kitti', dest='kitti',
                        help='Using KITTI dataset or not, otherwise defaults to Pascal VOC classes',
                        action='store_true')
    parser.add_argument('--phases', dest='phases',
                        help='Training phases, e.g. 60000:0.001,20000:0.0001 for'
                             '60k iterations with learning rate 0.001'
                             'followed by 20k iterations with learning rate 0.0001',
                        default="60000:1e-3,20000:1e-4")
    parser.add_argument('--optimizer', dest='optimizer', choices=('sgd', 'adam'),
                        help='sgd or adam',
                        default="sgd")
    parser.add_argument('--img_set', dest='img_set', choices=('train', 'val', 'trainval'),
                        help='which image set to use, must be one of train, val, or trainval',
                        default="trainval")
    parser.add_argument('--resize_dims', dest='resize_dims',
                        help='resize parameters, e.g. 600,1000 if resizing to a min size of 600 pixels and max 1000 pixels',
                        default="600,1000")
    parser.add_argument('--anchor_scales', dest='anchor_scales',
                        help='anchor scales in pixels, e.g. 128,256,512 if following the original paper',
                        default="128,256,512")
    parser.add_argument('--network', dest='network', choices=('vgg16', 'resnet50', 'resnet101'),
                        help='underlying network architecture, choose from vgg16, resnet50 or resnet101',
                        default="vgg16")
    parser.add_argument('--save_weights_dest', dest='save_weights_dest',
                        help='Location to save detector weights, should end in .h5',
                        default=None)
    parser.add_argument('--save_model_dest', dest='save_model_dest',
                        help='Location to save the detector model, should end in .h5',
                        default=None)
    parser.add_argument('--save_rpn_model_dest', dest='save_rpn_model_dest',
                        help='Location to save the step 3 rpn model, should end in .h5',
                        default=None)

    args = parser.parse_args()

    train_imgs = base_paths_to_imgs(args.voc_paths, img_set=args.img_set)
    anchor_scales = anchor_scales_from_str(args.anchor_scales)
    anchors = get_anchors(anchor_scales)
    anchors_per_loc = len(anchors)
    resize_min, resize_max = resize_dims_from_str(args.resize_dims)
    processed_imgs, resized_ratios = resize_imgs(train_imgs, min_size=resize_min, max_size=resize_max)
    phases = phases_from_str(args.phases)
    optimizer = optimizer_from_str(args.optimizer)
    print("num train_imgs: ", len(train_imgs))

    class_mapping = KITTI_CLASS_MAPPING if args.kitti else VOC_CLASS_MAPPING
    num_classes = len(class_mapping)
    if args.network == 'vgg16':
        # don't need to worry about freezing/regularizing rpn because we're not training it
        base_model = vgg.vgg16_base()
        rpn_model = vgg.vgg16_rpn(base_model, include_conv=True, anchors_per_loc=anchors_per_loc)
        preprocess_func = vgg.preprocess
        get_conv_rows_cols_func = vgg.get_conv_rows_cols
        stride = vgg.STRIDE
        detector_model = vgg.vgg16_classifier(NUM_ROIS, num_classes,
                                              weight_regularizer=vgg.WEIGHT_REGULARIZER,
                                              bias_regularizer=vgg.BIAS_REGULARIZER)
    elif args.network == 'resnet50':
        # don't need to worry about freezing/regularizing rpn because we're not training it
        base_model = resnet.resnet50_base()
        rpn_model = resnet.resnet50_rpn(base_model, include_conv=True, anchors_per_loc=anchors_per_loc)
        preprocess_func = resnet.preprocess
        get_conv_rows_cols_func = resnet.get_conv_rows_cols
        stride = resnet.STRIDE
        detector_model = resnet.resnet50_classifier(NUM_ROIS, num_classes, weight_regularizer=resnet.WEIGHT_REGULARIZER,
                                                    bias_regularizer=resnet.BIAS_REGULARIZER)
    elif args.network == 'resnet101':
        # don't need to worry about freezing/regularizing rpn because we're not training it
        base_model = resnet.resnet101_base()
        rpn_model = resnet.resnet101_rpn(base_model, include_conv=True, anchors_per_loc=anchors_per_loc)
        preprocess_func = resnet.preprocess
        get_conv_rows_cols_func = resnet.get_conv_rows_cols
        stride = resnet.STRIDE
        detector_model = resnet.resnet101_classifier(NUM_ROIS, num_classes,
                                                     weight_regularizer=resnet.WEIGHT_REGULARIZER,
                                                     bias_regularizer=resnet.BIAS_REGULARIZER)

    if args.save_weights_dest:
        save_weights_dest = args.save_weights_dest
    else:
        save_weights_dest = "models/detector_weights_{}_step4.h5".format(args.network)
    if args.save_model_dest:
        save_model_dest = args.save_model_dest
    else:
        save_model_dest = "models/detector_model_{}_step4.h5".format(args.network)
    if args.save_rpn_model_dest:
        save_rpn_model_dest = args.save_rpn_model_dest
    else:
        save_rpn_model_dest = "models/rpn_model_{}_step3.h5".format(args.network)

    rpn_model.load_weights(args.step3_weights_path)
    training_manager = DetTrainingManager(rpn_model=rpn_model, class_mapping=class_mapping,
                                          preprocess_func=preprocess_func, stride=stride, anchor_dims=anchors)
    train_detector_step4(detector_model, processed_imgs, training_manager, optimizer,
                         phases=phases, save_frequency=2000, save_weights_dest=save_weights_dest,
                         save_model_dest=save_model_dest)

    detector_model.save_weights(save_weights_dest)
    print('Saved {} detector weights to {}'.format(args.network, save_weights_dest))
    detector_model.save(save_model_dest)
    print('Saved {} detector model to {}'.format(args.network, save_model_dest))
    rpn_model.save(save_rpn_model_dest)
    print('Saved {} rpn model to {}'.format(args.network, save_rpn_model_dest))
