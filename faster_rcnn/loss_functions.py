import tensorflow as tf

from keras.objectives import categorical_crossentropy
from keras import backend as K

from shared_constants import DEFAULT_ANCHORS_PER_LOC

N_CLS = 256
N_REG = 2400

LAMBDA_REG = 10.0
LAMBDA_REG_DET = 1


def cls_loss_rpn(anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    """
    Creates a loss function for the object classifying output of the RPN module.
    :param anchors_per_loc: how many anchors at each convolution position.
    :return: a function used as a Keras loss function.
    """
    def cls_loss_rpn_internal(y_true, y_pred):
        selected_losses = y_true[:, :, :, :anchors_per_loc]
        y_is_pos = y_true[:, :, :, anchors_per_loc:]
        loss = K.sum(selected_losses * K.binary_crossentropy(y_is_pos, y_pred)) / N_CLS

        return loss

    return cls_loss_rpn_internal


def bbreg_loss_rpn(anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    """
    Creates a loss function for the bounding box regression output of the RPN. Uses the "smooth" loss function defined
    in the paper.
    :param anchors_per_loc: how many anchors at each convolution position.
    :return: a function used as a Keras loss function.
    """
    def bbreg_loss_rpn_internal(y_true, y_pred):
        selected_losses = y_true[:, :, :, :4 * anchors_per_loc]
        diff = y_true[:, :, :, 4 * anchors_per_loc:] - y_pred
        abs_diff = K.abs(diff)
        multipliers_small = K.cast(K.less_equal(abs_diff, 1.0), tf.float32)
        multipliers_big = 1.0 - multipliers_small
        loss = LAMBDA_REG * selected_losses * K.sum(multipliers_small * (0.5 * abs_diff * abs_diff) + multipliers_big * (abs_diff - 0.5)) / N_REG

        return loss

    return bbreg_loss_rpn_internal


def bbreg_loss_det(num_classes):
    """
    Creates a loss function for the "smooth" bounding box regression output of the Fast R-CNN module. See the paper
    for details.
    :param num_classes: positive integer, the number of object classes used, NOT including background.
    :return: a function used as a Keras loss function.
    """
    def class_loss_internal(y_true, y_pred):
        # diff for bb reg
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        # only consider loss from the ground truth class
        # use smooth L1 loss function
        loss = K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(1e-4 + y_true[:, :, :4*num_classes])
        return LAMBDA_REG_DET * loss
    return class_loss_internal


def cls_loss_det(y_true, y_pred):
    """
    Loss function for the object classification output of the Fast R-CNN module.
    :param num_classes: positive integer, the number of object classes used NOT including background.
    :return: tensor for the category cross entry loss.
    """
    return K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
