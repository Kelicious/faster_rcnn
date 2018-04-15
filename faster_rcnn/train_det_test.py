import numpy as np
import random
np.random.seed(1337)
random.seed(a=1)
import tensorflow as tf
tf.set_random_seed(42)

from keras.optimizers import Adam
import h5py
import unittest
import os
from subprocess import Popen, PIPE

import vgg
import resnet
from vgg import vgg16_rpn, vgg16_base, vgg16_classifier
from resnet import resnet50_base, resnet50_rpn, resnet50_classifier
from train_util import train_detector_step2
from util import resize_imgs, get_anchors
from data.voc_data_helpers import extract_img_data, VOC_CLASS_MAPPING
from det_util import DetTrainingManager

NUM_ROIS = 64


class TrainFrcnnCase(unittest.TestCase):

    def test_vgg16_frcnn_training_phase_2(self):
        # setup
        anchors = get_anchors(anchor_scales=[128, 256, 512])
        anchors_per_loc = len(anchors)
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        test_dir = os.path.join(cur_dir, os.pardir, 'test_data')
        base_dir = os.path.join(test_dir, 'VOC_test')
        ref_weights_path = os.path.join(test_dir, 'reference_frcnn_step2_weights.h5')
        tmp_weights_path = os.path.join(test_dir, 'tmp_frcnn_weights.h5')
        rpn_weights_path = os.path.join(test_dir, 'rpn_weights_step1.h5')
        img = extract_img_data(base_dir, '000005')
        training_imgs, resized_ratios = resize_imgs([img])

        model_rpn = vgg16_rpn(vgg16_base(), anchors_per_loc=anchors_per_loc)
        model_rpn.load_weights(filepath=rpn_weights_path)
        model_frcnn = vgg16_classifier(num_rois=64, num_classes=21, base_model=vgg16_base())

        class_mapping = VOC_CLASS_MAPPING
        training_manager = DetTrainingManager(rpn_model=model_rpn, class_mapping=class_mapping, num_rois=NUM_ROIS,
                                              preprocess_func=vgg.preprocess, anchor_dims=anchors)
        optimizer = Adam(lr=0.001)

        # action being tested
        train_detector_step2(detector=model_frcnn, images=training_imgs, training_manager=training_manager,
                             optimizer=optimizer, phases=[[1, 0.001]])

        # assertion
        last_layer_weights = model_frcnn.get_layer('block5_conv3').get_weights()[0]
        with h5py.File(tmp_weights_path, 'w') as file:
            file.create_dataset('last_layer_weights', data=last_layer_weights)
        process = Popen(['h5diff', ref_weights_path, tmp_weights_path], stdout=PIPE, stderr=PIPE)
        process.communicate()
        self.assertEqual(process.returncode, 0)

    def test_resnet_frcnn_training_phase_2(self):
        # setup
        anchors = get_anchors(anchor_scales=[128, 256, 512])
        anchors_per_loc = len(anchors)
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        test_dir = os.path.join(cur_dir, os.pardir, 'test_data')
        base_dir = os.path.join(test_dir, 'VOC_test')
        ref_weights_path = os.path.join(test_dir, 'reference_r50_frcnn_step2_weights.h5')
        tmp_weights_path = os.path.join(test_dir, 'tmp_r50_frcnn_weights.h5')
        rpn_weights_path = os.path.join(test_dir, 'r50_rpn_step1.h5')
        img = extract_img_data(base_dir, '000005')
        training_imgs, resized_ratios = resize_imgs([img])

        model_rpn = resnet50_rpn(resnet50_base(), anchors_per_loc=anchors_per_loc)
        model_rpn.load_weights(filepath=rpn_weights_path)
        model_frcnn = resnet50_classifier(num_rois=64, num_classes=21, base_model=resnet50_base())

        class_mapping = VOC_CLASS_MAPPING
        training_manager = DetTrainingManager(rpn_model=model_rpn, class_mapping=class_mapping, num_rois=NUM_ROIS,
                                              preprocess_func=resnet.preprocess, anchor_dims=anchors)
        optimizer = Adam(lr=0.001)

        # action being tested
        train_detector_step2(detector=model_frcnn, images=training_imgs, training_manager=training_manager,
                             optimizer=optimizer, phases=[[1, 0.0001]])

        # assertion
        last_layer_weights = model_frcnn.get_layer('res5c_branch2c').get_weights()[0]
        with h5py.File(tmp_weights_path, 'w') as file:
            file.create_dataset('last_layer_weights', data=last_layer_weights)
        process = Popen(['h5diff', ref_weights_path, tmp_weights_path], stdout=PIPE, stderr=PIPE)
        process.communicate()
        self.assertEqual(process.returncode, 0)


if __name__ == '__main__':
    unittest.main()
