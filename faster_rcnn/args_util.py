from keras.optimizers import SGD, Adam

from data.voc_data_helpers import extract_img_data, get_img_names_from_set
from shared_constants import DEFAULT_LEARN_RATE, DEFAULT_MOMENTUM


def base_paths_to_imgs(base_path_str, img_set='trainval', do_flip=True):
    """
    Parses a command line argument containing one or multiple locations of training/inference images.
    :param base_path_str: string, contains absolute filesystem paths separated by commas. Each path should point to the
    root directory of an image set formatted according to the PASCAL VOC directory structure.
    :param img_set: string, one of 'train', 'val', 'trainval', or 'test'.
    :param do_flip: boolean, whether to include horizontally flipped copies of the images. Used for training but not
    inference.
    :return: list of shapes.Image objects.
    """
    paths = base_path_str.split(',')
    imgs = []
    for path in paths:
        img_names = get_img_names_from_set(path, img_set)
        curr_imgs = [extract_img_data(path, img_name) for img_name in img_names]
        imgs.extend(curr_imgs)

    if do_flip:
        flipped_imgs = [img.horizontal_flip() for img in imgs]
        imgs += flipped_imgs
    return imgs


def phases_from_str(phases_str):
    """
    Parses a command line argument string describing the learning rate schedule for training.
    :param phases_str: string formatted like 60000:1e-3,20000:1e-4 for 60k iterations with learning rate 1e-3 followed
    by 20k iterations with learning rate 1e-4.
    :return: list of lists of an integer and floating point number pair, e.g. "60000:1e-3,20000:1e-4" returns
     [[60000, 1e-3], [20000, 1e-4]]
    """
    parts = phases_str.split(',')
    phases = []
    for phase_str in parts:
        splits = phase_str.split(':')
        iterations, learning_rate = int(splits[0]), float(splits[1])
        phases.append([iterations, learning_rate])

    return phases


def optimizer_from_str(optimizer_str):
    """
    Parses a command line argument string for the optimizer.
    :param optimizer_str: 'adam' or 'sgd'.
    :return: keras.optimizers.SGD instance with momentum 0.9 if the argument was 'sgd', keras.optimizers.Adam instance
     if the argument was 'adam'.
    """
    # initial learn rate is ignored and set at runtime based on the phases
    if optimizer_str == 'sgd':
        return SGD(lr=DEFAULT_LEARN_RATE, momentum=DEFAULT_MOMENTUM)
    else:
        return Adam(lr=DEFAULT_LEARN_RATE)


def resize_dims_from_str(resize_dims_str):
    """
    Parses a command line argument string for the resize parameters.
    :param resize_dims_str: comma-separated integers, e.g. "600,1000".
    :return: list of integers.
    """
    return [int(dim) for dim in resize_dims_str.split(',')]


def anchor_scales_from_str(anchor_scales_str):
    """
    Parses a command line argument string for the anchor scales.
    :param anchor_scales_str: comma-separated integers, e.g. "128,256,512".
    :return: list of integers.
    """
    return [int(dim) for dim in anchor_scales_str.split(',')]
