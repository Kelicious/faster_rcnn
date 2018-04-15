import cv2
import numpy as np

from keras import regularizers
from keras.applications import vgg16
from keras.applications.vgg16 import VGG16, WEIGHTS_PATH_NO_TOP
from keras.initializers import TruncatedNormal
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, load_model
from keras.utils.data_utils import get_file

from custom_layers import RoiResizeConv
from loss_functions import cls_loss_det, bbreg_loss_det, cls_loss_rpn, bbreg_loss_rpn
from shared_constants import DEFAULT_ANCHORS_PER_LOC

POOLING_REGIONS = 7
FINAL_CONV_FILTERS = 512
STRIDE = 16
WEIGHT_REGULARIZER = None #regularizers.l2(5e-4)
BIAS_REGULARIZER = None #regularizers.l2(5e-4)
# not sure if activity regularizer is needed anywhere
ACTIVITY_REGULARIZER = regularizers.l2(5e-4)


def rpn_from_h5(h5_path, anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    model_rpn = load_model(h5_path,
                           custom_objects={'cls_loss_rpn': cls_loss_rpn(anchors_per_loc=anchors_per_loc),
                                           'bbreg_loss_rpn': bbreg_loss_rpn(anchors_per_loc=anchors_per_loc)})

    return model_rpn


def det_from_h5(h5_path, num_classes):
    """
    Loads a saved detector model from an h5 file.
    :param h5_path: string, filesystem path of the saved Keras model for the detector module.
    :param num_classes: positive integer, the number of object classes (including background) used in the file's model.
    :return: Keras model.
    """
    model_det = load_model(h5_path,
                           custom_objects={'RoiResizeConv': RoiResizeConv,
                                           'cls_loss_det': cls_loss_det,
                                           'bbreg_loss_det': bbreg_loss_det,
                                           'class_loss_internal': bbreg_loss_det(num_classes)})

    return model_det


def preprocess(data):
    rgb_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    batched_rgb_data = np.expand_dims(rgb_data, axis = 0).astype('float64')
    new_data = vgg16.preprocess_input(batched_rgb_data)[0]

    return new_data


def get_conv_rows_cols(height, width):
    return height // STRIDE, width // STRIDE


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        # hack to set model.outputs properly
        model.outputs = [model.layers[-1].output]
        # another hack to set model.output properly
        model.inbound_nodes[0].output_tensors[-1] = model.outputs[-1]
    model.built = False


def vgg16_base_old():
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)

    # remove last max pooling layer
    pop_layer(base_model)
    
    return base_model


def vgg16_base(freeze_blocks=[1, 2], weight_regularizer=None, bias_regularizer=None):
    img_input = Input(shape=(None, None, 3))

    # Block 1
    train1 = 1 not in freeze_blocks
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=train1,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=train1,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    train2 = 2 not in freeze_blocks
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=train2,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=train2,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    train3 = 3 not in freeze_blocks
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=train3,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=train3,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=train3,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    train4 = 4 not in freeze_blocks
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=train4,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=train4,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=train4,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    train5 = 5 not in freeze_blocks
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=train5,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=train5,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=train5,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)

    base_model = Model(img_input, x, name='resnet50')

    return base_model


def vgg16_rpn_old(base_model, include_conv=False, weight_regularizer=None, bias_regularizer=None, anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    for i in range(6):
        base_model.layers[i].trainable = False

    net = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_model.output)

    gaussian_initializer = TruncatedNormal(stddev=0.01)
    x_class = Conv2D(anchors_per_loc, (1, 1), activation='sigmoid',
                     kernel_initializer=gaussian_initializer,
                     kernel_regularizer=weight_regularizer,
                     bias_regularizer=bias_regularizer,
                     name='rpn_out_cls')(net)
    x_regr = Conv2D(anchors_per_loc * 4, (1, 1), activation='linear',
                    kernel_initializer=gaussian_initializer,
                    kernel_regularizer=weight_regularizer,
                    bias_regularizer=bias_regularizer,
                    name='rpn_out_bbreg')(net)

    outputs = [x_class, x_regr]
    if include_conv:
        outputs.append(base_model.output)
    rpn_model = Model(inputs = base_model.inputs, outputs = outputs)
    return rpn_model


def vgg16_rpn(base_model, include_conv=False, weight_regularizer=None, bias_regularizer=None,
              anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    gaussian_initializer = TruncatedNormal(stddev=0.01)
    net = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=gaussian_initializer,
                 kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                 name='rpn_conv1', )(base_model.output)

    x_class = Conv2D(anchors_per_loc, (1, 1), activation='sigmoid',
                     kernel_initializer=gaussian_initializer,
                     kernel_regularizer=weight_regularizer,
                     bias_regularizer=bias_regularizer,
                     name='rpn_out_cls')(net)
    x_regr = Conv2D(anchors_per_loc * 4, (1, 1), activation='linear',
                    kernel_initializer=gaussian_initializer,
                    kernel_regularizer=weight_regularizer,
                    bias_regularizer=bias_regularizer,
                    name='rpn_out_bbreg')(net)

    outputs = [x_class, x_regr]
    if include_conv:
        outputs.append(base_model.output)
    rpn_model = Model(inputs = base_model.inputs, outputs = outputs)
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            file_hash='6d6bbae143d832006294945121d1f1fc')
    rpn_model.load_weights(weights_path, by_name=True)
    return rpn_model


def vgg16_classifier_old(num_rois, num_classes, base_model = None, weight_regularizer=None, bias_regularizer=None):
    roi_input = Input(shape=(None, 4), name='roi_input')

    pooling_input = base_model.output if base_model else Input(shape=(None, None, FINAL_CONV_FILTERS))
    model_input = base_model.input if base_model else pooling_input
    roi_pooling_out = RoiResizeConv(POOLING_REGIONS, num_rois)([pooling_input, roi_input])

    out = TimeDistributed(Flatten(name='flatten'))(roi_pooling_out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1',
                                kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2',
                                kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer))(out)

    gaussian_initializer_class = TruncatedNormal(stddev=0.01)
    gaussian_initializer_reg = TruncatedNormal(stddev=0.001)

    out_class = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=gaussian_initializer_class,
                                      kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer),
                                name='dense_class_{}'.format(num_classes))(out)
    out_reg = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear', kernel_initializer=gaussian_initializer_reg,
                                    kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer),
                              name='dense_reg_{}'.format(num_classes))(out)

    cls_model = Model(inputs=[model_input, roi_input], outputs=[out_class, out_reg])
    return cls_model


def vgg16_classifier(num_rois, num_classes, base_model = None, weight_regularizer=None, bias_regularizer=None):
    roi_input = Input(shape=(None, 4), name='roi_input')

    pooling_input = base_model.output if base_model else Input(shape=(None, None, FINAL_CONV_FILTERS))
    model_input = base_model.input if base_model else pooling_input
    roi_pooling_out = RoiResizeConv(POOLING_REGIONS, num_rois)([pooling_input, roi_input])

    out = TimeDistributed(Flatten(name='flatten'))(roi_pooling_out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1',
                                kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2',
                                kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer))(out)

    gaussian_initializer_class = TruncatedNormal(stddev=0.01)
    gaussian_initializer_reg = TruncatedNormal(stddev=0.001)

    out_class = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=gaussian_initializer_class,
                                      kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer),
                                name='dense_class_{}'.format(num_classes))(out)
    out_reg = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear', kernel_initializer=gaussian_initializer_reg,
                                    kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer),
                              name='dense_reg_{}'.format(num_classes))(out)

    cls_model = Model(inputs=[model_input, roi_input], outputs=[out_class, out_reg])
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            file_hash='6d6bbae143d832006294945121d1f1fc')
    cls_model.load_weights(weights_path, by_name=True)
    return cls_model


VGG16_LAYERS = ['block1_conv1',
              'block1_conv2',
              'block1_pool',
              'block2_conv1',
              'block2_conv2',
              'block2_pool',
              'block3_conv1',
              'block3_conv2',
              'block3_conv3',
              'block3_pool',
              'block4_conv1',
              'block4_conv2',
              'block4_conv3',
              'block4_pool',
              'block5_conv1',
              'block5_conv2',
              'block5_conv3']


VGG16_TUNABLE_LAYERS = ['block1_conv1',
                           'block1_conv2',
                           'block2_conv1',
                           'block2_conv2',
                           'block3_conv1',
                           'block3_conv2',
                           'block3_conv3',
                           'block4_conv1',
                           'block4_conv2',
                           'block4_conv3',
                           'block5_conv1',
                           'block5_conv2',
                        'block5_conv3']
