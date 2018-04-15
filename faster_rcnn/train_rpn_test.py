import random
import unittest
import os
from subprocess import Popen, PIPE

from keras.optimizers import Adam
import h5py
import numpy as np
import vgg

from vgg import vgg16_rpn, vgg16_base
from data.voc_data_helpers import extract_img_data
from rpn_util import RpnTrainingManager
from train_util import train_rpn
from util import get_anchors

np.random.seed(1337)
random.seed(a=1)


class TrainRpnCase(unittest.TestCase):
    def test_rpn_training(self):
        # setup
        anchors = get_anchors(anchor_scales=[128, 256, 512])
        anchors_per_loc = len(anchors)
        model_rpn = vgg16_rpn(vgg16_base(), anchors_per_loc=anchors_per_loc)
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        test_dir = os.path.join(cur_dir, os.pardir, 'test_data')
        base_dir = os.path.join(test_dir, 'VOC_test')
        ref_weights_path = os.path.join(test_dir, 'reference_rpn_weights.h5')
        tmp_weights_path = os.path.join(test_dir, 'tmp_rpn_weights.h5')
        image = extract_img_data(base_dir, '000005')
        training_manager = RpnTrainingManager(vgg.get_conv_rows_cols, vgg.STRIDE, preprocess_func=vgg.preprocess,
                                              anchor_dims=anchors)
        optimizer = Adam(lr=0.001)

        # action being tested
        train_rpn(model_rpn, [image], training_manager, optimizer, phases=[[1, 0.001]])

        # assertion
        last_layer_weights = model_rpn.get_layer('block5_conv3').get_weights()[0]
        with h5py.File(tmp_weights_path, 'w') as file:
            file.create_dataset('last_layer_weights', data=last_layer_weights)
        process = Popen(['h5diff', ref_weights_path, tmp_weights_path], stdout=PIPE, stderr=PIPE)
        process.communicate()
        self.assertEqual(process.returncode, 0)


if __name__ == '__main__':
    unittest.main()
