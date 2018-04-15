import random
import timeit

from keras import backend as K

from loss_functions import cls_loss_rpn, bbreg_loss_rpn, cls_loss_det, bbreg_loss_det
from shared_constants import DEFAULT_NUM_ITERATIONS, DEFAULT_LEARN_RATE


def train_rpn(rpn_model, images, training_manager, optimizer, phases=[[DEFAULT_NUM_ITERATIONS, DEFAULT_LEARN_RATE]],
              save_frequency=None, save_weights_dest=None, save_model_dest=None):
    """
    Trains a region proposal network.
    :param rpn_model: Keras model for the rpn to be trained.
    :param images: sequence of shapes.Image objects used to train the network.
    :param training_manager: rpn_util.RpnTrainingManager to produce training inputs from images.
    :param optimizer: keras.optimizers.Optimizer implementation to be used. Doesn't need a preconfigured learning rate.
    :param phases: list of lists specifying the learning rate schedule, e.g. [[1000, 1e-3], [100, 1e-4]] 1000 iterations
    with learning rate 1e-3 followed by 100 iterations with learning rate 1e-4.
    :param save_frequency: positive integer specifying how many iterations occur between saving the model's state. Leave
    it as None to disable saving during training.
    :param save_weights_dest: the path to save model weights as an h5 file after each save_frequency iterations.
    :param save_model_dest: the path to save the Keras model as an h5 file after each save_frequency iterations.
    :return: the rpn passed in.
    """
    num_train = len(images)

    anchors_per_loc = len(training_manager.anchor_dims)
    for phase_num, phase in enumerate(phases):
        num_iterations, learn_rate = phase
        optimizer.lr = K.variable(learn_rate, name='lr')
        rpn_model.compile(optimizer=optimizer, loss=[cls_loss_rpn(anchors_per_loc=anchors_per_loc),
                                                     bbreg_loss_rpn(anchors_per_loc=anchors_per_loc)])

        print("Starting phase {} of training: {} iterations with learning rate {}".format(
            phase_num, num_iterations, learn_rate))

        for i in range(num_iterations):
            img_idx = (i + num_iterations * phase_num) % num_train
            if img_idx == 0:
                random.shuffle(images)

            img = images[img_idx]

            print('Starting phase {} iteration {} with learn rate {}, training on image {}, flipped status: {}'.format(
                phase_num, i, learn_rate, img.name, img.flipped))

            print('img size: {}x{}'.format(img.width, img.height))

            batched_img = training_manager.batched_image(img)
            y_class, y_bbreg = training_manager.rpn_y_true(img)

            start_time = timeit.default_timer()
            loss_rpn = rpn_model.train_on_batch(batched_img, [y_class, y_bbreg])
            print("model_rpn.train_on_batch time: ", timeit.default_timer() - start_time)
            print('loss_rpn: {}'.format(loss_rpn))

            if save_frequency and i % save_frequency == 0:
                if save_weights_dest != None:
                    rpn_model.save_weights(save_weights_dest)
                    print('Saved rpn weights to {}'.format(save_weights_dest))
                if save_model_dest != None:
                    rpn_model.save(save_model_dest)
                    print('Saved rpn model to {}'.format(save_model_dest))

    return rpn_model


def train_detector_step2(detector, images, training_manager, optimizer,
                         phases=[[DEFAULT_NUM_ITERATIONS, DEFAULT_LEARN_RATE]], save_frequency=None,
                         save_weights_dest=None, save_model_dest=None):
    """
    Trains a Fast R-CNN object detector for step 2 of the 4-step alternate training scheme in the paper.
    :param detector: Keras model for the detector module used in step 2 of training. The model should accepts images
    and regions as inputs.
    :param images: sequence of shapes.Image objects used to train the network.
    :param training_manager: det_util.DetTrainingManager object produce training inputs from images.
    :param optimizer: keras.optimizers.Optimizer implementation to be used. Doesn't need a preconfigured learning rate.
    :param phases: list of lists specifying the learning rate schedule, e.g. [[1000, 1e-3], [100, 1e-4]] 1000 iterations
    with learning rate 1e-3 followed by 100 iterations with learning rate 1e-4.
    :param save_frequency: positive integer specifying how many iterations occur between saving the model's state. Leave
    it as None to disable saving during training.
    :param save_weights_dest: the path to save model weights as an h5 file after each save_frequency iterations. Leave
    this as None to disable weight saving during training.
    :param save_model_dest: the path to save the Keras model as an h5 file after each save_frequency iterations. Leave
    this as None to disable model saving during training.
    :return: the detector passed in.
    """
    num_train = len(images)
    num_classes = len(training_manager.class_mapping) - 1

    for phase_num, phase in enumerate(phases):
        num_iterations, learn_rate = phase
        optimizer.lr = K.variable(learn_rate, name='lr')
        detector.compile(optimizer=optimizer, loss=[cls_loss_det, bbreg_loss_det(num_classes)])

        print("Starting phase {} of training: {} iterations with learning rate {}".format(
            phase_num, num_iterations, learn_rate))

        for i in range(num_iterations):
            img_idx = (i + num_iterations * phase_num) % num_train
            if img_idx == 0:
                random.shuffle(images)

            img = images[img_idx]

            print('Starting phase {} iteration {} with learn rate {}, training on image {}, flipped status: {}'.format(
                phase_num, i, learn_rate, img.name, img.flipped))

            batched_img, rois, y_class_num, y_transform = training_manager.get_training_input(img)

            if rois is None:
                print("Found no rois for this image")
                continue

            import timeit
            start_time = timeit.default_timer()
            loss_frcnn = detector.train_on_batch([batched_img, rois], [y_class_num, y_transform])
            print("model_frcnn.train_on_batch time: ", timeit.default_timer() - start_time)
            print('loss_frcnn: {}'.format(loss_frcnn))

            if save_frequency and i > 0 and i % save_frequency == 0:
                if save_weights_dest != None:
                    detector.save_weights(save_weights_dest)
                    print('Saved detector weights to {}'.format(save_weights_dest))
                if save_model_dest != None:
                    detector.save(save_model_dest)
                    print('Saved detector model to {}'.format(save_model_dest))

    return detector


def train_detector_step4(detector, images, training_manager, optimizer,
                         phases=[[DEFAULT_NUM_ITERATIONS, DEFAULT_LEARN_RATE]], save_frequency=None,
                         save_weights_dest=None, save_model_dest=None):
    """
    Trains a Fast R-CNN object detector for step 4 of the 4-step alternate training scheme in the paper.
    :param detector: Keras model for the detector module used in step 4 of training. The module should accept images'
    convolutional features and regions as inputs.
    :param images: sequence of shapes.Image objects used to train the network.
    :param training_manager: det_util.DetTrainingManager object produce training inputs from images.
    :param optimizer: keras.optimizers.Optimizer implementation to be used. Doesn't need a preconfigured learning rate.
    :param phases: list of lists specifying the learning rate schedule, e.g. [[1000, 1e-3], [100, 1e-4]] 1000 iterations
    with learning rate 1e-3 followed by 100 iterations with learning rate 1e-4.
    :param save_frequency: positive integer specifying how many iterations occur between saving the model's state. Leave
    it as None to disable saving during training.
    :param save_weights_dest: the path to save model weights as an h5 file after each save_frequency iterations. Leave
    this as None to disable weight saving during training.
    :param save_model_dest: the path to save the Keras model as an h5 file after each save_frequency iterations. Leave
    this as None to disable model saving during training.
    :return: the detector passed in.
    """
    num_train = len(images)
    num_classes = len(training_manager.class_mapping) - 1

    for phase_num, phase in enumerate(phases):
        num_iterations, learn_rate = phase
        optimizer.lr = K.variable(learn_rate, name='lr')
        detector.compile(optimizer=optimizer, loss=[cls_loss_det, bbreg_loss_det(num_classes)])

        print("Starting phase {} of training: {} iterations with learning rate {}".format(
            phase_num, num_iterations, learn_rate))

        for i in range(num_iterations):
            img_idx = (i + num_iterations * phase_num) % num_train
            if img_idx == 0:
                random.shuffle(images)

            img = images[img_idx]

            print('Starting phase {} iteration {} with learn rate {}, training on image {}, flipped status: {}'.format(
                phase_num, i, learn_rate, img.name, img.flipped))

            conv_features, rois, y_class_num, y_transform = training_manager.get_training_input(img)

            if rois is None:
                print("Found no training samples for this image")
                continue

            import timeit
            start_time = timeit.default_timer()
            loss_frcnn = detector.train_on_batch([conv_features, rois], [y_class_num, y_transform])
            print("model_frcnn.train_on_batch time: ", timeit.default_timer() - start_time)
            print('loss_frcnn: {}'.format(loss_frcnn))

            if save_frequency and i > 0 and i % save_frequency == 0:
                if save_weights_dest != None:
                    detector.save_weights(save_weights_dest)
                    print('Saved detector weights to {}'.format(save_weights_dest))
                if save_model_dest != None:
                    detector.save(save_model_dest)
                    print('Saved detector model to {}'.format(save_model_dest))

    return detector