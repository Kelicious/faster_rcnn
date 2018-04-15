import os

import cv2
import keras.backend as K
import keras.layers as layers
import numpy as np
from keras import regularizers
from keras.applications import resnet50
from keras.applications.resnet50 import WEIGHTS_PATH_NO_TOP
from keras.initializers import TruncatedNormal
from keras.layers import Input, BatchNormalization, Activation, AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten, Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, load_model
from keras.utils.data_utils import get_file

from custom_layers import RoiResizeConv, Scale
from loss_functions import cls_loss_det, bbreg_loss_det, cls_loss_rpn, bbreg_loss_rpn
from shared_constants import DEFAULT_ANCHORS_PER_LOC

POOLING_REGIONS = 7
FINAL_CONV_FILTERS = 1024
STRIDE = 16

WEIGHT_REGULARIZER = regularizers.l2(1e-4)
BIAS_REGULARIZER = regularizers.l2(1e-4)
# not sure if activity regularizer is needed anywhere
ACTIVITY_REGULARIZER = regularizers.l2(1e-4)


def rpn_from_h5(h5_path, anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    """
    Loads a saved rpn model from an h5 file.
    :param h5_path: string, filesystem path of the saved Keras model for the rpn.
    :param anchors_per_loc: positive integer, the number of used in the rpn saved in the file.
    :return: Keras model.
    """
    model_rpn = load_model(h5_path,
                           custom_objects={'cls_loss_rpn': cls_loss_rpn(anchors_per_loc=anchors_per_loc),
                                           'bbreg_loss_rpn': bbreg_loss_rpn(anchors_per_loc=anchors_per_loc),
                                           'Scale': Scale})

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
                                           'Scale': Scale,
                                           'cls_loss_det': cls_loss_det,
                                           'bbreg_loss_det': bbreg_loss_det,
                                           'class_loss_internal': bbreg_loss_det(num_classes)})

    return model_det


def preprocess(data):
    """
    Convert raw bgr image to the format needed for pre-trained Imagenet weights to apply.
    :param data: numpy array containing bgr values of an image.
    :return: numpy array with preprocessed values.
    """
    # expect image to be passed in as BGR
    rgb_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    batched_rgb_data = np.expand_dims(rgb_data, axis = 0).astype('float64')
    new_data = resnet50.preprocess_input(batched_rgb_data)[0]

    return new_data


def get_conv_rows_cols(height, width):
    """
    Calculates the dimensions of the last conv4 layer for a given image size.
    :param height: positive integer, the image height in pixels.
    :param width: positive integer, the image width in pixels.
    :return: height and width of the last conv4 layer as a list of integers.
    """
    dims = [height, width]
    for i in range(len(dims)):
        # (3, 3) zeropad
        dims[i] += 6
        for filter_size in [7, 3, 1, 1]:
            # all strides use valid padding, formula is (W - F + 2P) / S + 1
            dims[i] = (dims[i] - filter_size) // 2 + 1

    return dims


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


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True, use_conv_bias=True,
                   weight_regularizer=None, bias_regularizer=None, bn_training=False, separate_scale=False):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Additional arguments
        trainable: boolean for whether to make this block's layers trainable.
        use_conv_bias: boolean for whether or not convolutional layers should have a bias.
        weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
        regularization.
        bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
        bn_training: boolean for whether or not BatchNormalization layers should be trained. Should always be false as
        the model doesn't train correctly with batch normalization.
        separate_scale: boolean for whether or not the BatchNormalization layers should be followed by a separate Scale
        layer.

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    eps = 1e-5

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', trainable=trainable, use_bias=use_conv_bias,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a',
                           trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2a', trainable=bn_training)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable,
               use_bias=use_conv_bias,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b',
                           trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2b', trainable=bn_training)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', trainable=trainable, use_bias=use_conv_bias,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c',
                           trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2c', trainable=bn_training)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True, use_conv_bias=True,
               weight_regularizer=None, bias_regularizer=None, bn_training=False, separate_scale=False):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Additional arguments
        trainable: boolean for whether to make this block's layers trainable.
        use_conv_bias: boolean for whether or not convolutional layers should have a bias.
        weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
        regularization.
        bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
        bn_training: boolean for whether or not BatchNormalization layers should be trained. Should always be false as
        the model doesn't train correctly with batch normalization.
        separate_scale: boolean for whether or not the BatchNormalization layers should be followed by a separate Scale
        layer.

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    eps = 1e-5

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable, use_bias=use_conv_bias,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a', trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2a', trainable=bn_training)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable,
               use_bias=use_conv_bias, kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b', trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2b', trainable=bn_training)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', trainable=trainable, use_bias=use_conv_bias,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c', trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2c', trainable=bn_training)(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable, use_bias=use_conv_bias,
                      kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1',
                                  trainable=bn_training)(shortcut, training=bn_training)
    if separate_scale:
        shortcut = Scale(axis=bn_axis, name=scale_name_base + '1', trainable=bn_training)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def td_identity_block(input_tensor, kernel_size, filters, stage, block, use_conv_bias=True,
                      weight_regularizer=None, bias_regularizer=None, bn_training=False, separate_scale=False):
    """Time distributed version of resnet identity block

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Additional arguments
        use_conv_bias: boolean for whether or not convolutional layers should have a bias.
        weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
        regularization.
        bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
        bn_training: boolean for whether or not BatchNormalization layers should be trained. Should always be false as
        the model doesn't train correctly with batch normalization.
        separate_scale: boolean for whether or not the BatchNormalization layers should be followed by a separate Scale
        layer.

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    eps = 1e-5

    x = TimeDistributed(Conv2D(filters1, (1, 1), use_bias=use_conv_bias,
                               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                               ),
                        name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(epsilon=eps, axis=bn_axis, trainable=bn_training),
                        name=bn_name_base + '2a',)(x, training=bn_training)
    if separate_scale:
        x = TimeDistributed(Scale(axis=bn_axis, trainable=bn_training), name=scale_name_base + '2a')(x, training=bn_training)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(filters2, kernel_size, padding='same', use_bias=use_conv_bias,
                               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer
                               ),
                        name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(epsilon=eps, axis=bn_axis, trainable=bn_training),
                        name=bn_name_base + '2b')(x, training=bn_training)
    if separate_scale:
        x = TimeDistributed(Scale(axis=bn_axis, trainable=bn_training), name=scale_name_base + '2b')(x, training=bn_training)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(filters3, (1, 1), use_bias=use_conv_bias,
                               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer
                               ),
                        name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(epsilon=eps, axis=bn_axis, trainable=bn_training),
                        name=bn_name_base + '2c')(x, training=bn_training)
    if separate_scale:
        x = TimeDistributed(Scale(axis=bn_axis, trainable=bn_training), name=scale_name_base + '2c')(x, training=bn_training)

    x = layers.add([x, input_tensor])
    x = TimeDistributed(Activation('relu'))(x)
    return x


def td_conv_block(input_tensor, kernel_size, filters, stage, block, td_input_shape, strides=(2, 2), use_conv_bias=True,
                  weight_regularizer=None, bias_regularizer=None, bn_training=False, separate_scale=False):
    """A time distributed block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Additional arguments
        use_conv_bias: boolean for whether or not convolutional layers should have a bias.
        weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
        regularization.
        bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
        bn_training: boolean for whether or not BatchNormalization layers should be trained. Should always be false as
        the model doesn't train correctly with batch normalization.
        separate_scale: boolean for whether or not the BatchNormalization layers should be followed by a separate Scale
        layer.

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    eps = 1e-5

    x = TimeDistributed(Conv2D(filters1, (1, 1), strides=strides, use_bias=use_conv_bias,
                                    kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer
                               ),
                        name=conv_name_base + '2a', input_shape=td_input_shape)(input_tensor)
    x = TimeDistributed(BatchNormalization(epsilon=eps, axis=bn_axis, trainable=bn_training),
                        name=bn_name_base + '2a')(x, training=bn_training)
    if separate_scale:
        x = TimeDistributed(Scale(axis=bn_axis, trainable=bn_training), name=scale_name_base + '2a')(x, training=bn_training)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(filters2, kernel_size, padding='same', use_bias=use_conv_bias,
                                    kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer
                               ),
                        name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(epsilon=eps, axis=bn_axis, trainable=bn_training),
                        name=bn_name_base + '2b')(x, training=bn_training)
    if separate_scale:
        x = TimeDistributed(Scale(axis=bn_axis, trainable=bn_training), name=scale_name_base + '2b')(x, training=bn_training)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(filters3, (1, 1), use_bias=use_conv_bias,
                                    kernel_regularizer=weight_regularizer,bias_regularizer=bias_regularizer
                               ),
                        name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(epsilon=eps, axis=bn_axis, trainable=bn_training),
                        name=bn_name_base + '2c')(x, training=bn_training)
    if separate_scale:
        x = TimeDistributed(Scale(axis=bn_axis, trainable=bn_training), name=scale_name_base + '2c')(x, training=bn_training)

    shortcut = TimeDistributed(Conv2D(filters3, (1, 1), strides=strides, use_bias=use_conv_bias,
                                    kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer
                                    ),
                               name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(epsilon=eps, axis=bn_axis, trainable=bn_training),
                               name=bn_name_base + '1')(shortcut, training=bn_training)
    if separate_scale:
        shortcut = TimeDistributed(Scale(axis=bn_axis, trainable=bn_training),
                                   name=scale_name_base + '1')(shortcut, training=bn_training)

    x = layers.add([x, shortcut])
    x = TimeDistributed(Activation('relu'))(x)
    return x


def resnet50_base(freeze_blocks=[1,2,3], weight_regularizer=None, bias_regularizer=None):
    """
    Creates a model of the ResNet-50 base layers used for both the RPN and detector.
    :param freeze_blocks: list of block numbers to make untrainable, e.g. [1,2,3] to not train the first 3 blocks.
    :param weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
    regularization.
    :param bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
    :return: Keras model for the base network.
    """
    img_input = Input(shape=(None, None, 3))
    bn_axis = 3
    train1 = 1 not in freeze_blocks
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', trainable=train1,
        kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', trainable=False)(x, training=False)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    train2 = 2 not in freeze_blocks
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=train2,
                   weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=train2,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=train2,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)

    train3 = 3 not in freeze_blocks
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=train3,
                   weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=train3,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=train3,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=train3,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)

    train4 = 4 not in freeze_blocks
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=train4,
                   weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=train4,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=train4,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=train4,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=train4,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=train4,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)

    base_model = Model(img_input, x, name='resnet50')

    return base_model


def resnet50_rpn(base_model, weight_regularizer=None, bias_regularizer=None, include_conv=False,
                 anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    """
    Creates an rpn model on top of a passed in base model.
    :param base_model: Keras model returned by resnet50_base, containing only the first 4 blocks.
    :param weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
    regularization.
    :param bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
    :param include_conv: boolean for whether the conv4 output should be included in the model output.
    :param anchors_per_loc: number of anchors at each convolution position.
    :return: Keras model with the rpn layers on top of the base layers. Weights are initialized to Imagenet weights.
    """
    net = Conv2D(512, (3, 3), padding='same', activation='relu',kernel_initializer='normal',
                 kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                 name='rpn_conv1')(base_model.output)

    gaussian_initializer = TruncatedNormal(stddev=0.01)
    x_class = Conv2D(anchors_per_loc, (1, 1), activation='sigmoid', kernel_initializer=gaussian_initializer,
                     kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                     name='rpn_out_cls')(net)
    x_regr = Conv2D(anchors_per_loc * 4, (1, 1), activation='linear', kernel_initializer=gaussian_initializer,
                    kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                    name='rpn_out_bbreg')(net)

    outputs = [x_class, x_regr]
    if include_conv:
        outputs.append(base_model.output)

    rpn_model = Model(inputs = base_model.inputs, outputs = outputs)
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            md5_hash='a268eb855778b3df3c7506639542a6af')
    rpn_model.load_weights(weights_path, by_name=True)
    return rpn_model


def resnet50_classifier(num_rois, num_classes, base_model = None, weight_regularizer=None, bias_regularizer=None):
    """
    Creates a Keras model of the ResNet-50 classification layers on top of a passed in base model.
    :param num_rois: positive integer, number of regions of interest to train or inference on in a batch.
    :param num_classes: positive integer, number of object classes including background.
    :param base_model: Keras model returned by resnet50_base, containing only the first 4 blocks.
    :param weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
    regularization.
    :param bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
    :return: Keras model with the classification layers on top of the base layers. Weights are initialized to Imagenet
    weights.
    """
    roi_input = Input(shape=(None, 4), name='roi_input')

    pooling_input = base_model.output if base_model else Input(shape=(None, None, FINAL_CONV_FILTERS))
    model_input = base_model.input if base_model else pooling_input
    resize_out = RoiResizeConv(POOLING_REGIONS, num_rois)([pooling_input, roi_input])

    out = td_conv_block(resize_out, 3, [512, 512, 2048], stage=5, block='a', strides=(1,1),
                        weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                        td_input_shape=(num_rois, POOLING_REGIONS, POOLING_REGIONS, 1024))
    out = td_identity_block(out, 3, [512, 512, 2048], stage=5, block='b',
                            weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    out = td_identity_block(out, 3, [512, 512, 2048], stage=5, block='c',
                            weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    out = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(out)

    out = TimeDistributed(Flatten(name='flatten'))(out)

    gaussian_initializer_cls = TruncatedNormal(stddev=0.01)
    gaussian_initializer_bbreg = TruncatedNormal(stddev=0.001)

    out_class = TimeDistributed(Dense(num_classes, activation='softmax',
                                      kernel_initializer=gaussian_initializer_cls,
                                      kernel_regularizer=weight_regularizer,
                                      bias_regularizer=bias_regularizer
                                      ),
                                      name='dense_class_{}'.format(num_classes))(out)
    out_reg = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear',
                                    kernel_initializer=gaussian_initializer_bbreg,
                                    kernel_regularizer=weight_regularizer,
                                    bias_regularizer=bias_regularizer
                                    ),
                                    name='dense_reg_{}'.format(num_classes))(out)

    cls_model = Model(inputs=[model_input, roi_input], outputs=[out_class, out_reg])

    # not sure if needed - bn layers should already be frozen
    for layer in cls_model.layers:
        if isinstance(layer, TimeDistributed) and isinstance(layer.layer, BatchNormalization):
            layer.layer.trainable = False

    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            md5_hash='a268eb855778b3df3c7506639542a6af')
    cls_model.load_weights(weights_path, by_name=True)

    return cls_model


def resnet101_base(freeze_blocks=[1,2,3], weight_regularizer=None, bias_regularizer=None):
    """
    Creates a model of the ResNet-101 base layers used for both the RPN and detector.
    :param freeze_blocks: list of block numbers to make untrainable, e.g. [1,2,3] to not train the first 3 blocks.
    :param weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
    regularization.
    :param bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
    :return: Keras model for the base network.
    """
    img_input = Input(shape=(None, None, 3))
    bn_axis = 3
    train1 = 1 not in freeze_blocks
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', trainable=train1, use_bias=False,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', trainable=False)(x, training=False)
    x = Scale(axis=bn_axis, name='scale_conv1', trainable=False)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    train2 = 2 not in freeze_blocks
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=train2,
                   weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                   use_conv_bias=False, separate_scale=True)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=train2,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                   use_conv_bias=False, separate_scale=True)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=train2,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                   use_conv_bias=False, separate_scale=True)

    train3 = 3 not in freeze_blocks
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=train3,
                   weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                   use_conv_bias=False, separate_scale=True)
    for i in range(1, 4):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i), trainable=train3,
                           weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                   use_conv_bias=False, separate_scale=True)

    train4 = 4 not in freeze_blocks
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=train4,
                   weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                   use_conv_bias=False, separate_scale=True)
    for i in range(1, 23):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i), trainable=train4,
                           weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                   use_conv_bias=False, separate_scale=True)

    base_model = Model(img_input, x, name='resnet101')

    return base_model


def resnet101_rpn(base_model, weight_regularizer=None, bias_regularizer=None, include_conv=False,
                  anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    # like resnet50_rpn but loads a different set of weights
    net = Conv2D(512, (3, 3), padding='same', activation='relu',kernel_initializer='normal',
                 kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                 name='rpn_conv1')(base_model.output)

    gaussian_initializer = TruncatedNormal(stddev=0.01)
    x_class = Conv2D(anchors_per_loc, (1, 1), activation='sigmoid', kernel_initializer=gaussian_initializer,
                     kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                     name='rpn_out_cls')(net)
    x_regr = Conv2D(anchors_per_loc * 4, (1, 1), activation='linear', kernel_initializer=gaussian_initializer,
                    kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                    name='rpn_out_bbreg')(net)

    outputs = [x_class, x_regr]
    if include_conv:
        outputs.append(base_model.output)

    rpn_model = Model(inputs = base_model.inputs, outputs = outputs)
    this_dir = os.path.dirname(__file__)
    weights_path = os.path.join(this_dir, '../models/resnet101_weights_tf.h5')
    rpn_model.load_weights(weights_path, by_name=True)
    return rpn_model


def resnet101_classifier(num_rois, num_classes, base_model = None, weight_regularizer=None, bias_regularizer=None):
    """
    Creates a Keras model of the ResNet-101 classification layers on top of a passed in base model.
    :param num_rois: positive integer, number of regions of interest to train or inference on in a batch.
    :param num_classes: positive integer, number of object classes including background.
    :param base_model: Keras model returned by resnet101_base, containing only the first 4 blocks.
    :param weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
    regularization.
    :param bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
    :return: Keras model with the classification layers on top of the base layers. Weights are initialized to Imagenet
    weights.
    """
    roi_input = Input(shape=(None, 4), name='roi_input')

    pooling_input = base_model.output if base_model else Input(shape=(None, None, FINAL_CONV_FILTERS))
    model_input = base_model.input if base_model else pooling_input
    resize_out = RoiResizeConv(POOLING_REGIONS, num_rois)([pooling_input, roi_input])

    out = td_conv_block(resize_out, 3, [512, 512, 2048], stage=5, block='a', strides=(1,1),
                        weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                        td_input_shape=(num_rois, POOLING_REGIONS, POOLING_REGIONS, 1024),
                        use_conv_bias=False, separate_scale=True)
    out = td_identity_block(out, 3, [512, 512, 2048], stage=5, block='b',
                            weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                            use_conv_bias=False, separate_scale=True)
    out = td_identity_block(out, 3, [512, 512, 2048], stage=5, block='c',
                            weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                            use_conv_bias=False, separate_scale=True)
    out = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(out)

    out = TimeDistributed(Flatten(name='flatten'))(out)

    gaussian_initializer_cls = TruncatedNormal(stddev=0.01)
    gaussian_initializer_bbreg = TruncatedNormal(stddev=0.001)

    out_class = TimeDistributed(Dense(num_classes, activation='softmax',
                                      kernel_initializer=gaussian_initializer_cls,
                                      kernel_regularizer=weight_regularizer,
                                      bias_regularizer=bias_regularizer
                                      ),
                                      name='dense_class_{}'.format(num_classes))(out)
    out_reg = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear',
                                    kernel_initializer=gaussian_initializer_bbreg,
                                    kernel_regularizer=weight_regularizer,
                                    bias_regularizer=bias_regularizer
                                    ),
                                    name='dense_reg_{}'.format(num_classes))(out)

    cls_model = Model(inputs=[model_input, roi_input], outputs=[out_class, out_reg])

    this_dir = os.path.dirname(__file__)
    weights_path = os.path.join(this_dir, '../models/resnet101_weights_tf.h5')
    cls_model.load_weights(weights_path, by_name=True)

    return cls_model
