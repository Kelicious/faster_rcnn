from keras.engine.topology import Layer, InputSpec
from keras import initializers
import keras.backend as K
import tensorflow as tf


class RoiResizeConv(Layer):
    """
    Implements the region of interest max pooling layer. Crops the region instead of max pooling it.
    """

    def __init__(self, pool_size, num_rois, **kwargs):
        """
        Just a constructor.
        :param pool_size: dimension of one side of the output, i.e. 7 to pool convolutional features into a 7x7 output.
        :param num_rois: number of regions in one mini-batch.
        :param kwargs:
        """
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiResizeConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

        super(RoiResizeConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def get_config(self):
        return {'pool_size': self.pool_size, 'num_rois': self.num_rois}

    def call(self, x):
        img = x[0]
        rois = x[1]

        outputs = []

        for roi_idx in range(self.num_rois):
            roi = rois[0, roi_idx]
            x1, y1, x2, y2 = roi[0], roi[1], roi[2], roi[3]

            x1 = K.cast(x1, 'int32')
            y1 = K.cast(y1, 'int32')
            x2 = K.cast(x2, 'int32')
            y2 = K.cast(y2, 'int32')

            rs = tf.image.resize_images(img[:, y1:y2, x1:x2, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return final_output


class Scale(Layer):
    '''Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
    '''
    def __init__(self,
                 weights=None,
                 axis=-1,
                 momentum=0.9,
                 beta_init='zero',
                 gamma_init='one',
                 **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(
            self.gamma_init(shape),
            name='{}_gamma'.format(self.name))
        self.beta = K.variable(
            self.beta_init(shape),
            name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(
            self.gamma,
            broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))