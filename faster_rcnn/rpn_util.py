import random
from enum import Enum

import numpy as np

from custom_decorators import profile
from shapes import Box
from shared_constants import BBREG_MULTIPLIERS, DEFAULT_ANCHORS
from util import calc_iou, cross_ious, get_reg_params, get_bbox_coords

POS_OVERLAP = 0.7
NEG_OVERLAP = 0.3

SAMPLE_SIZE = 256
MAX_POS_SAMPLES = 128


class RpnClass(Enum):
    NEG = 0
    POS = 1
    NEUTRAL = 2


class RpnTrainingManager:
    """
    Encapsulates the details of generating training inputs for a region proposal network for a given image.
    """

    def __init__(self, calc_conv_dims, stride, preprocess_func, anchor_dims=DEFAULT_ANCHORS):
        """
        :param calc_conv_dims: function that accepts a tuple of the image's height and width in pixels and returns the
        height and width of the convolutional layer prior to the rpn layers.
        :param stride: positive integer, the cumulative stride at the convolutional layer prior to the rpn layers.
        :param preprocess_func: function that applies the same transformation to the image's pixels as used for Imagenet
        training. Otherwise the Imagenet pre-trained weights will be mismatched.
        :param anchor_dims: list of lists of positive integers, one height and width pair for each anchor.
        """
        self._cache = {}
        self.calc_conv_dims = calc_conv_dims
        self.stride = stride
        self.preprocess_func = preprocess_func
        self.anchor_dims = anchor_dims

    @profile
    def batched_image(self, image):
        """
        Returns the image data to be fed into the network.
        :param image: shapes.Image object.
        :return: 4-d numpy array with a single batch of the image, should can be used as a Keras model input.
        """
        return np.expand_dims(self.preprocess_func(image.data), axis=0)

    @profile
    def _process(self, image):
        # internal method, performs the expensive calculations needed to produce training inputs.
        conv_rows, conv_cols = self.calc_conv_dims(image.height, image.width)
        num_anchors = conv_rows * conv_cols * len(self.anchor_dims)
        bbreg_targets = np.zeros((num_anchors, 4), dtype=np.float32)
        can_use = np.zeros(num_anchors, dtype=np.bool)
        is_pos = np.zeros(num_anchors, dtype=np.bool)

        gt_box_coords = get_bbox_coords(image.gt_boxes)

        anchor_coords = _get_all_anchor_coords(conv_rows, conv_cols, self.anchor_dims, self.stride)
        out_of_bounds_idxs = _get_out_of_bounds_idxs(anchor_coords, image.width, image.height)
        all_ious = cross_ious(anchor_coords, gt_box_coords)
        # all_ious, out_of_bounds_idxs = get_all_ious_faster(gt_box_coords, conv_rows, conv_cols, ANCHORS_PER_LOC, image.width, image.height, self.stride)

        max_iou_by_anchor = np.amax(all_ious, axis=1)
        max_idx_by_anchor = np.argmax(all_ious, axis=1)
        max_iou_by_gt_box = np.amax(all_ious, axis=0)
        max_idx_by_gt_box = np.argmax(all_ious, axis=0)

        # anchors with more than 0.7 IOU with a gt box are positives
        pos_box_idxs = np.where(max_iou_by_anchor > POS_OVERLAP)[0]
        # for each gt box, the highest non-zero IOU anchor is a positive
        eligible_idxs = np.where(max_iou_by_gt_box > 0.0)
        more_pos_box_idxs = max_idx_by_gt_box[eligible_idxs]

        total_pos_idxs = np.unique(np.concatenate((pos_box_idxs, more_pos_box_idxs)))
        can_use[total_pos_idxs] = 1
        is_pos[total_pos_idxs] = 1

        # don't bother optimizing, profiling showed this loop's runtime is negligible
        for box_idx in total_pos_idxs:
            y, x, anchor_idx = _idx_to_conv(box_idx, conv_cols, len(self.anchor_dims))
            x_center, y_center = _get_conv_center(x, y, self.stride)
            anchor_height, anchor_width = self.anchor_dims[anchor_idx]
            anchor_box = Box.from_center_dims_int(x_center, y_center, anchor_width, anchor_height)
            gt_box_idx = max_idx_by_anchor[box_idx]

            reg_params = get_reg_params(anchor_box.corners, gt_box_coords[gt_box_idx])
            bbreg_targets[box_idx, :] = BBREG_MULTIPLIERS * reg_params

        neg_box_idxs = np.where(np.logical_and(is_pos == 0, max_iou_by_anchor < NEG_OVERLAP))[0]
        can_use[neg_box_idxs] = 1
        can_use[out_of_bounds_idxs] = 0

        self._cache[image.cache_key] = {
            'can_use': can_use,
            'is_pos': is_pos,
            'bbreg_targets': bbreg_targets
        }

    @profile
    def rpn_y_true(self, image):
        """
        Takes an image and returns the Keras model inputs to train with.
        :param image: shapes.Image object to generate training inputs for.
        :return: tuple where the first element is a numpy array of the ground truth network output for whether each
        anchor overlaps with an object, and the second element is a numpy array of the ground truth network output for the
        bounding box transformation parameters to transform each anchor into an object's bounding box.
        """
        '''
        Consider removing caching - added when self.process was taking 0.4s to run. Since then, optimized it down to
        0.02s locally, 0.003s on aws so the cache isn't too useful anymore.
        '''
        if image.cache_key not in self._cache:
            self._process(image)

        results = self._cache[image.cache_key]
        # TODO: why is the cached result being deleted? Investigate whether restoring it improves training time.
        del self._cache[image.cache_key]
        can_use = _apply_sampling(results['is_pos'], results['can_use'])
        conv_rows, conv_cols = self.calc_conv_dims(image.height, image.width)

        is_pos = np.reshape(results['is_pos'], (conv_rows, conv_cols, len(self.anchor_dims)))
        can_use = np.reshape(can_use, (conv_rows, conv_cols, len(self.anchor_dims)))
        selected_is_pos = np.logical_and(is_pos, can_use)

        # combine arrays with whether or not to use for the loss function
        y_class = np.concatenate([can_use, is_pos], axis=2)
        bbreg_can_use = np.repeat(selected_is_pos, 4, axis = 2)
        bbreg_targets = np.reshape(results['bbreg_targets'], (conv_rows, conv_cols, 4 * len(self.anchor_dims)))
        y_bbreg = np.concatenate([bbreg_can_use, bbreg_targets], axis = 2)

        y_class = np.expand_dims(y_class, axis=0)
        y_bbreg = np.expand_dims(y_bbreg, axis=0)

        return y_class, y_bbreg


def _idx_to_conv(idx, conv_width, anchors_per_loc):
    """
    Converts an anchor box index in a 1-d numpy array to its corresponding 3-d index representing its convolution
    position and anchor index.
    :param idx: non-negative integer, the position in a 1-d numpy array of anchors.
    :param conv_width: the number of possible horizontal positions the convolutional layer's filters can occupy, i.e.
    close to the width in pixels divided by the cumulative stride at that layer.
    :param anchors_per_loc: positive integer, the number of anchors at each convolutional filter position.
    :return: tuple of the row, column, and anchor index of the convolutional filter position for this index.
    """
    divisor = conv_width * anchors_per_loc
    y, remainder = idx // divisor, idx % divisor
    x, anchor_idx = remainder // anchors_per_loc, remainder % anchors_per_loc
    return y, x, anchor_idx


@profile
def _num_boxes_to_conv_np(num_boxes, conv_width, anchors_per_loc):
    # similar to _idx_to_conv but for multiple boxes at once, uses vectorized operations to optimize the performance
    idxs = np.arange(num_boxes)
    divisor = conv_width * anchors_per_loc
    y, remainder = idxs // divisor, idxs % divisor
    x, anchor_idx = remainder // anchors_per_loc, remainder % anchors_per_loc
    return y, x, anchor_idx


def _get_conv_center(conv_x, conv_y, stride):
    """
    Finds the center of this convolution position in the image's original coordinate space.
    :param conv_x: non-negative integer, x coordinate of the convolution position.
    :param conv_y: non-negative integer, y coordinate of the convolution position.
    :param stride: positive integer, the cumulative stride in pixels at this layer of the network.
    :return: tuple of positive integers, the x and y coordinates of the center of the convolution position.
    """
    x_center = stride * (conv_x + 0.5)
    y_center = stride * (conv_y + 0.5)

    return int(x_center), int(y_center)


@profile
def _get_conv_center_np(conv_x, conv_y, stride):
    # like _get_conv_center but optimized for multiple boxes.
    x_center = stride * (conv_x + 0.5)
    y_center = stride * (conv_y + 0.5)

    return x_center.astype('int32'), y_center.astype('int32')


@profile
def _get_all_ious(bbox_coords, conv_rows, conv_cols, anchor_dims, img_width, img_height, stride):
    # not used anymore, might be useful to keep around as a reference
    num_boxes = conv_rows * conv_cols * len(anchor_dims)
    num_gt_boxes = len(bbox_coords)
    result = np.zeros((num_boxes, num_gt_boxes))
    out_of_bounds_idxs = []

    num_boxes = conv_rows * conv_cols * len(anchor_dims)
    for i in range(num_boxes):
        y, x, anchor_idx = _idx_to_conv(i, conv_cols, len(anchor_dims))
        x_center, y_center = _get_conv_center(x, y, stride)
        anchor_height, anchor_width = anchor_dims[anchor_idx]
        anchor_box = Box.from_center_dims_int(x_center, y_center, anchor_width, anchor_height)

        if _out_of_bounds(anchor_box, img_width, img_height):
            out_of_bounds_idxs.append(i)
            continue

        for bbox_idx in range(num_gt_boxes):
            iou = calc_iou(bbox_coords[bbox_idx], anchor_box.corners)

            result[i, bbox_idx] = iou

    return result, out_of_bounds_idxs


@profile
def _get_all_ious_fast(bbox_coords, conv_rows, conv_cols, anchor_dims, img_width, img_height, stride):
    # optimization of _get_all_ious using vectorized operations, also not used anymore
    num_boxes = conv_rows * conv_cols * len(anchor_dims)
    num_gt_boxes = len(bbox_coords)
    result = np.zeros((num_boxes, num_gt_boxes))
    out_of_bounds_idxs = []

    num_boxes = conv_rows * conv_cols * len(anchor_dims)
    coords = np.zeros((4))
    for i in range(num_boxes):
        y, x, anchor_idx = _idx_to_conv(i, conv_cols, len(anchor_dims))
        x_center, y_center = _get_conv_center(x, y, stride)
        anchor_height, anchor_width = anchor_dims[anchor_idx]
        coords[0] = x_center - anchor_width // 2
        coords[2] = coords[0] + anchor_width
        coords[1] = y_center - anchor_height // 2
        coords[3] = coords[1] + anchor_height

        if _out_of_bounds_coords(coords, img_width, img_height):
            out_of_bounds_idxs.append(i)
            continue

        for bbox_idx in range(num_gt_boxes):
            iou = calc_iou(bbox_coords[bbox_idx], coords)

            result[i, bbox_idx] = iou

    return result, out_of_bounds_idxs


@profile
# this function was a huge bottleneck so threw away box abstractions to optimize performance
def _get_all_ious_faster(bbox_coords, conv_rows, conv_cols, anchor_dims, img_width, img_height, stride):
    # even more optimized version of _get_all_ious_fast, also not used anymore
    num_boxes = conv_rows * conv_cols * len(anchor_dims)

    y, x, anchor_idxs = _num_boxes_to_conv_np(num_boxes, conv_cols, len(anchor_dims))
    x_center, y_center = _get_conv_center_np(x, y, stride)
    anchor_coords = np.zeros((num_boxes, 4))
    anchor_height = anchor_dims[anchor_idxs, 0]
    anchor_width = anchor_dims[anchor_idxs, 1]
    anchor_coords[:, 0] = x_center - anchor_width // 2
    anchor_coords[:, 1] = y_center - anchor_height // 2
    anchor_coords[:, 2] = anchor_coords[:, 0] + anchor_width
    anchor_coords[:, 3] = anchor_coords[:, 1] + anchor_height

    result = cross_ious(anchor_coords, bbox_coords)
    out_of_bounds_idxs = np.where(np.logical_or.reduce((
        anchor_coords[:,0] < 0,
        anchor_coords[:,1] < 0,
        anchor_coords[:,2] >= img_width,
        anchor_coords[:,3] >= img_height)))[0]
    return result, out_of_bounds_idxs


@profile
def _get_all_anchor_coords(conv_rows, conv_cols, anchor_dims, stride):
    """
    Given the shape of a convolutional layer and the anchors to generate for each position, return all anchors.
    :param conv_rows: positive integer, height of this convolutional layer.
    :param conv_cols: positive integer, width of this convolutional layer.
    :param anchor_dims: list of lists of positive integers, one height and width pair for each anchor.
    :param stride: positive integer, cumulative stride of this anchor position in pixels.
    :return: 2-d numpy array with one row for each anchor box containing its [x1, y1, x2, y2] coordinates.
    """
    num_boxes = conv_rows * conv_cols * len(anchor_dims)

    y, x, anchor_idxs = _num_boxes_to_conv_np(num_boxes, conv_cols, len(anchor_dims))
    x_center, y_center = _get_conv_center_np(x, y, stride)
    anchor_coords = np.zeros((num_boxes, 4), dtype=np.float32)
    anchor_height = anchor_dims[anchor_idxs, 0]
    anchor_width = anchor_dims[anchor_idxs, 1]

    anchor_coords[:, 0] = x_center - anchor_width // 2
    anchor_coords[:, 1] = y_center - anchor_height // 2
    anchor_coords[:, 2] = anchor_coords[:, 0] + anchor_width
    anchor_coords[:, 3] = anchor_coords[:, 1] + anchor_height

    return anchor_coords


@profile
def _get_out_of_bounds_idxs(anchor_coords, img_width, img_height):
    # internal function for figuring out which anchors are out of bounds
    out_of_bounds_idxs = np.where(np.logical_or.reduce((
        anchor_coords[:,0] < 0,
        anchor_coords[:,1] < 0,
        anchor_coords[:,2] >= img_width,
        anchor_coords[:,3] >= img_height)))[0]

    return out_of_bounds_idxs


def _out_of_bounds(box, width, height):
    # internal function for checking whether a box is out of bounds, not used anymore
    return box.x1 < 0 or box.x2 >= width or box.y1 < 0 or box.y2 >= height


def _out_of_bounds_coords(coords, width, height):
    # similar to _out_of_bounds but takes its argument as a numpy array instead of a shapes.Box instance
    return coords[0] < 0 or coords[2] >= width or coords[1] < 0 or coords[3] >= height


@profile
def _apply_sampling(is_pos, can_use):
    """
    Applies the sampling logic described in the Faster R-CNN paper to determine which anchors should be evaluated in the
    loss function.
    :param is_pos: 1-d numpy array of booleans for whether each anchor is a true positive for some object.
    :param can_use: 1-d numpy array of booleans for whether each anchor can be used at all in the loss function.
    :return: 1-d numpy array of booleans of which anchors were chosen to be used in the loss function.
    """
    # extract [0] due to np.where returning a tuple
    pos_locs = np.where(np.logical_and(is_pos == 1, can_use == 1))[0]
    neg_locs = np.where(np.logical_and(is_pos == 0, can_use == 1))[0]

    num_pos = len(pos_locs)
    num_neg = len(neg_locs)

    # cap the number of positive samples per batch to no more than half the batch size
    if num_pos > MAX_POS_SAMPLES:
        locs_off = random.sample(range(num_pos), num_pos - MAX_POS_SAMPLES)
        can_use[pos_locs[locs_off]] = 0
        num_pos = MAX_POS_SAMPLES

    # fill remaining portion of the batch size with negative samples
    if num_neg + num_pos > SAMPLE_SIZE:
        locs_off = random.sample(range(num_neg), num_neg + num_pos - SAMPLE_SIZE)
        can_use[neg_locs[locs_off]] = 0

    return can_use
