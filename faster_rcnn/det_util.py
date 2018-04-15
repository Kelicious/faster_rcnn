import numpy as np

from custom_decorators import profile
from shared_constants import BBREG_MULTIPLIERS, DEFAULT_ANCHORS
from util import transform_np_inplace, cross_ious, get_reg_params, get_bbox_coords

CLASSIFIER_MIN_OVERLAP = 0.1
CLASSIFIER_POS_OVERLAP = 0.5

PROBABLE_THRESHOLD = 0.05


class DetTrainingManager:
    """
    Encapsulates the details of generating training inputs for the Fast-RCNN module for a given image and a set of
    regions.
    """

    def __init__(self, rpn_model, class_mapping, preprocess_func, num_rois=64, stride=16, anchor_dims=DEFAULT_ANCHORS):
        self.rpn_model = rpn_model
        self.class_mapping = class_mapping
        self.preprocess_func = preprocess_func
        self.num_rois = num_rois
        self.stride = stride
        self.anchor_dims = anchor_dims
        self._cache = {}
        self.conv_only = True if len(rpn_model.output) == 3 else False

    @profile
    def batched_image(self, image):
        """
        Returns the image data to be fed into the network.
        :param image: shapes.Image object.
        :return: 4-d numpy array with a single batch of the image, should can be used as a Keras model input.
        """
        return np.expand_dims(self.preprocess_func(image.data), axis=0)

    @profile
    def _out_from_image(self, batched_img):
        # gets rpn output on a batched image
        return self.rpn_model.predict_on_batch(batched_img)

    @profile
    def _rois_from_image(self, image):
        # for a given image, return the relevant rpn outputs
        batched_img = self.batched_image(image)
        conv_out = None
        if self.conv_only:
            cls_out, regr_out, conv_out = self._out_from_image(batched_img)
        else:
            cls_out, regr_out = self._out_from_image(batched_img)
        roi_coords = self._get_rois(regr_out, self.anchor_dims)
        roi_probs = cls_out.reshape((-1))

        return roi_coords, roi_probs, conv_out

    @profile
    def _get_rois(self, regr_out, anchor_dims):
        # turn the rpn's regression output and anchor dimensions into regions
        return _get_rois(regr_out, anchor_dims, self.stride)

    @profile
    def _process(self, image):
        # internal method, performs the expensive calculations needed to produce training inputs.
        roi_coords, roi_probs, conv_out = self._rois_from_image(image)

        # TODO: get rid of this? already sanitized during previous step
        valid_idxs = _get_valid_box_idxs(roi_coords)
        roi_coords, roi_probs = roi_coords[valid_idxs], roi_probs[valid_idxs]
        # TODO: filtering out improbable ROIs would speed up NMS significantly, check if it hurts training results
        sorted_idxs = roi_probs.argsort()[::-1]
        # decreasing the number of boxes improves nms compute time
        truncated_idxs = sorted_idxs[0:12000]
        roi_coords, roi_probs = roi_coords[truncated_idxs], roi_probs[truncated_idxs]
        # casting to short ints cuts nms compute time by ~25%
        roi_coords = roi_coords.astype('int16')
        nms_rois, _ = nms(roi_coords, roi_probs, max_boxes=2000, overlap_thresh=0.7)
        filtered_rois, y_class_num, y_transform = _rois_to_truth(nms_rois, image, self.class_mapping, stride=self.stride)

        cache_obj = {
            'rois': filtered_rois,
            'y_class_num': y_class_num,
            'y_transform': y_transform
        }
        if conv_out is not None:
            cache_obj['conv_out'] = conv_out
        self._cache[image.cache_key] = cache_obj

    @profile
    def get_training_input(self, image):
        """
        Takes an image and returns the Keras model inputs to train with.
        :param image: shapes.Image object for which to generate training inputs.
        :return: tuple of 4 elements:
        1. The first input to the model. For step 2 of training it's the image's pixels after preprocessing. For step 4
        it's the convolutional features output by the last layer prior to the RPN module.
        2. 2-d numpy array containing the regions of interest selected for training. One row per region, formatted as
        [x1, y1, x2, y2] in coordinate space of the last convolutional layer prior to the RPN module.
        3. 2-d numpy array containing the one hot encoding of the object classes corresponding to each selected region.
        One row for each region containing a 1 in the column corresponding to the object class, 0 in all other columns.
        4. 3-d numpy array representing separate 2-d arrays:
          4a. one hot encoding of the object class for each selected region but with 4 copies of each number. This is
          used to determine which of the outputs in 4b should contribute to the loss function.
          4b. bounding box regression targets for each non-background object class, for each selected region. If there
          are 64 regions and 20 object classes, then this would be a 64 row numpy array with 80 columns, where columns
          0 through 3 inclusive contain the regression targets for object class 0, 4 through 7 inclusive contain the
          regression targets for object class 1, etc.
        """
        if image.cache_key not in self._cache:
            self._process(image)

        results = self._cache[image.cache_key]

        if len(results['rois']) == 0:
            return None, None, None, None

        rois, y_class_num, y_transform = results['rois'], results['y_class_num'], results['y_transform']
        # in the one-hot encoding the last index is bg hence use it to check neg/pos
        found_object = y_class_num[:, -1] == 0
        sampled_idxs = _get_det_samples(found_object, self.num_rois)

        rois, y_class_num, y_transform = rois[sampled_idxs], y_class_num[sampled_idxs], y_transform[sampled_idxs]

        # feed in the image during step 2, feed in the saved conv features during step 4
        first_input = results['conv_out'] if self.conv_only else self.batched_image(image)
        # in step 4 of training, caching the conv features takes too much memory so don't cache anything
        if self.conv_only:
            del self._cache[image.cache_key]

        return first_input,\
               np.expand_dims(rois, axis=0),\
               np.expand_dims(y_class_num, axis=0),\
               np.expand_dims(y_transform, axis=0)

    @profile
    def get_det_inputs(self, image):
        """
        Find the inputs to the step 4 Faster R-CNN module for a given image.
        :param image: shapes.Image object for which to generate training inputs.
        :return: tuple of two numpy arrays:
        1. features output by the last convolutional layer prior to the RPN module.
        2. regions of interest, each row containing the [x1, y1, x2, y2] from the coordinate space of the last
        convolutional layer prior to the RPN module.
        """
        roi_coords, roi_probs, conv_out = self._rois_from_image(image)

        valid_idxs = _get_valid_box_idxs(roi_coords)
        roi_coords, roi_probs = roi_coords[valid_idxs], roi_probs[valid_idxs]
        # TODO: try filtering to only use probable idxs, might shave some time off of the nms
        probable_idxs = np.where(roi_probs >= PROBABLE_THRESHOLD)[0]
        sorted_idxs = roi_probs.argsort()[::-1]

        truncated_idxs = sorted_idxs[0:8000]
        roi_coords, roi_probs = roi_coords[truncated_idxs], roi_probs[truncated_idxs]
        roi_coords = roi_coords.astype('int16')
        nms_rois, _ = nms(roi_coords, roi_probs, max_boxes=300, overlap_thresh=0.7)

        return conv_out, nms_rois


@profile
def _get_anchor_coords(conv_rows, conv_cols, anchor_dims, multiplier=1):
    # get the [x1, y1, x2, y2] coordinates of anchors for each convolution position.
    coords = np.zeros((conv_rows, conv_cols, len(anchor_dims), 4), dtype=np.float32)
    x_center, y_center = np.meshgrid(np.arange(conv_cols), np.arange(conv_rows))

    for i, anchor in enumerate(anchor_dims):
        anchor_height, anchor_width = anchor * multiplier

        coords[:, :, i, 0] = x_center - anchor_width // 2
        coords[:, :, i, 1] = y_center - anchor_height // 2
        coords[:, :, i, 2] = coords[:, :, i, 0] + anchor_width
        coords[:, :, i, 3] = coords[:, :, i, 1] + anchor_height

    return coords


@profile
def _sanitize_boxes_inplace(conv_cols, conv_rows, coords):
    # clip portions of regions lying outside the image boundaries.

    # set minimum width/height to 1
    coords[:, 2] = np.maximum(coords[:, 0] + 1, coords[:, 2])
    coords[:, 3] = np.maximum(coords[:, 1] + 1, coords[:, 3])
    # x1 and y1 must be at least 0
    coords[:, 0] = np.maximum(0, coords[:, 0])
    coords[:, 1] = np.maximum(0, coords[:, 1])
    # x2 and y2 must be at most cols-1 and rows-1
    coords[:, 2] = np.minimum(conv_cols - 1, coords[:, 2])
    coords[:, 3] = np.minimum(conv_rows - 1, coords[:, 3])

    return coords


@profile
def _get_valid_box_idxs(boxes):
    # find the boxes with positive width and height.
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    valid_idxs = np.where((x2 > x1) & (y2 > y1))[0]

    return valid_idxs


@profile
def nms(boxes, probs, overlap_thresh=0.7, max_boxes=300):
    """
    Applies non-maximum suppression to a set of boxes and their probabilities of containing an object.
    :param boxes: 2-d numpy array, one row for each box containing its [x1, y1, x2, y2] coordinates.
    :param probs: 1-d numpy array of floating point numbers, probability that the box with this index is an object.
    :param overlap_thresh: floating point number, a fraction indicating the minimum overlap between two boxes needed to
    suppress the one with lower probability.
    :param max_boxes: positive integer, how many output boxes desired.
    :return: tuple of 2 numpy arrays, the selected boxes and their probabilities in the same format as the inputs.
    """

    if len(boxes) == 0:
        return []

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        x1_intersection = np.maximum(x1[i], x1[idxs[:last]])
        y1_intersection = np.maximum(y1[i], y1[idxs[:last]])
        x2_intersection = np.minimum(x2[i], x2[idxs[:last]])
        y2_intersection = np.minimum(y2[i], y2[idxs[:last]])

        w_intersection = np.maximum(0, x2_intersection - x1_intersection + 1)
        h_intersection = np.maximum(0, y2_intersection - y1_intersection + 1)

        area_intersection = w_intersection * h_intersection

        area_union = area[i] + area[idxs[:last]] - area_intersection
        overlap = area_intersection / area_union

        idxs = idxs[np.where(overlap <= overlap_thresh)[0]]

        if len(pick) >= max_boxes:
            break

    return boxes[pick], probs[pick]


@profile
def _get_det_samples(is_pos, num_desired_rois):
    """
    Applies the sampling logic described in the Fast R-CNN paper for one mini-batch: sample 64 regions of interest total
    of which 25% should be positive.
    """

    desired_pos = num_desired_rois // 4
    pos_samples = np.where(is_pos)
    neg_samples = np.where(np.logical_not(is_pos))

    if len(neg_samples) > 0:
        neg_samples = neg_samples[0]
    else:
        neg_samples = []

    if len(pos_samples) > 0:
        pos_samples = pos_samples[0]
    else:
        pos_samples = []

    if len(pos_samples) == 0:
        selected_pos_samples = []
    elif len(pos_samples) < desired_pos:
        # num_copies = desired_pos // len(pos_samples) + 1
        # selected_pos_samples = np.tile(pos_samples, num_copies)[:desired_pos].tolist()
        selected_pos_samples = pos_samples.tolist()
    else:
        selected_pos_samples = np.random.choice(pos_samples, desired_pos, replace=False).tolist()

    desired_neg = num_desired_rois - len(selected_pos_samples)

    if len(neg_samples) == 0:
        selected_neg_samples = []
    elif len(neg_samples) < desired_neg:
        # num_copies = desired_neg // len(neg_samples) + 1
        # selected_neg_samples = np.tile(neg_samples, num_copies)[:desired_neg].tolist()
        selected_neg_samples = np.random.choice(neg_samples, desired_neg, replace=True).tolist()
    else:
        selected_neg_samples = np.random.choice(neg_samples, desired_neg, replace=False).tolist()

    if len(selected_neg_samples) == 0 and len(pos_samples) > 0:
        num_copies = desired_neg // len(pos_samples) + 1
        selected_neg_samples = np.tile(pos_samples, num_copies)[:desired_neg].tolist()

    selected_samples = selected_pos_samples + selected_neg_samples

    return selected_samples


@profile
def _rois_to_truth(rois, image, class_mapping, stride=16):
    # for an image, some regions of interest, and object classes, find the ground truth prior to sampling
    gt_boxes = [gt_box.resize(1/stride) for gt_box in image.gt_boxes]
    gt_coords = get_bbox_coords(gt_boxes)

    all_ious = cross_ious(rois, gt_coords)
    max_iou_by_roi = np.amax(all_ious, axis=1)
    max_gt_by_roi = np.argmax(all_ious, axis=1)
    eligible_roi_idxs = np.where(max_iou_by_roi >= CLASSIFIER_MIN_OVERLAP)[0]
    eligible_rois = rois[eligible_roi_idxs]

    pos_idxs = np.where(max_iou_by_roi >= CLASSIFIER_POS_OVERLAP)[0]
    eligible_gt_boxes = [gt_boxes[max_gt_by_roi[i]] if i in pos_idxs else None for i in eligible_roi_idxs]
    obj_classes = [box.obj_cls if box else 'bg' for box in eligible_gt_boxes]

    encoded_labels = _one_hot_encode_cls(obj_classes, class_mapping)

    bbreg_labels_targets = _one_hot_encode_bbreg(
        eligible_rois,
        eligible_gt_boxes,
        is_pos=np.isin(eligible_roi_idxs,pos_idxs),
        class_mapping=class_mapping
    )

    return eligible_rois, encoded_labels, bbreg_labels_targets


@profile
def _one_hot_encode_bbreg(rois, gt_boxes, is_pos, class_mapping):
    # finds the one hot encoded input for the bounding box regression part of the Fast-RCNN module prior to sampling.
    num_classes = len(class_mapping) - 1

    targs = np.zeros((len(rois), 4 * num_classes), dtype=np.float32)
    labels = np.zeros((len(rois), 4 * num_classes), dtype=np.float32)

    for i, (roi, gt_box, pos) in enumerate(zip(rois, gt_boxes, is_pos)):
        if pos:
            class_idx = class_mapping[gt_box.obj_cls]
            labels[i, 4*class_idx:4*(class_idx+1)] = 1, 1, 1, 1

            tx, ty, tw, th = get_reg_params(roi, gt_box.corners)
            targs[i, 4*class_idx:4*(class_idx+1)] = tx, ty, tw, th
            targs[i, 4*class_idx:4*(class_idx+1)] *= BBREG_MULTIPLIERS

    return np.concatenate([labels, targs], axis=1)


@profile
def _one_hot_encode_cls(obj_classes, class_to_num):
    # finds the one hot encoded input for the region classification part of the Fast-RCNN module prior to sampling.
    num_classes = len(class_to_num)
    result = np.zeros((len(obj_classes), num_classes), dtype=np.int32)

    for i, obj_cls in enumerate(obj_classes):
        result[i, class_to_num[obj_cls]] = 1

    return result


@profile
def _get_rois(regr_out, anchor_dims, stride):
    # turn the rpn's regression output and anchor dimensions into regions
    conv_rows, conv_cols = regr_out.shape[1:3]

    anchor_coords = _get_anchor_coords(conv_rows, conv_cols, anchor_dims // stride).reshape((-1, 4))
    reg_targets = regr_out[0].reshape((-1, 4))
    rois = transform_np_inplace(anchor_coords, reg_targets / BBREG_MULTIPLIERS)

    _sanitize_boxes_inplace(conv_cols, conv_rows, rois)

    return rois
