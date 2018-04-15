import math

import numpy as np

from custom_decorators import profile
from shared_constants import RESIZE_MIN_SIZE, RESIZE_MAX_SIZE, DEFAULT_ANCHOR_RATIOS, DEFAULT_ANCHOR_SCALES


def calc_iou(coords1, coords2):
    """
    Calculates the "intersection over union" overlap of two boxes. Both inputs are numpy arrays in the form
    [x1, y1, x2, y2].
    :param coords1: coordinates of the first box
    :param coords2: coordinates of the second box
    :return: floating point representation of the IOU
    """
    intersection = _calc_intersection(coords1, coords2)
    if intersection <= 0:
        return 0.0

    union = _calc_union(coords1, coords2)

    return intersection * 1.0 / union


def _calc_intersection(coords1, coords2):
    # calculates the area of the intersection only
    inter_x1 = max(coords1[0], coords2[0])
    inter_y1 = max(coords1[1], coords2[1])
    inter_x2 = min(coords1[2], coords2[2])
    inter_y2 = min(coords1[3], coords2[3])

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0

    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)


def _calc_union(coords1, coords2):
    # calculates the area of the union only
    area_1 = _area(coords1)
    area_2 = _area(coords2)

    return area_1 + area_2 - _calc_intersection(coords1, coords2)


def _area(coords):
    # calculates the area of a numpy box
    width = coords[2] - coords[0]
    height = coords[3] - coords[1]

    return width * height


def transform(anchor_coords, reg_targets):
    """
    Applies the bounding box regression transformation to an anchor box.
    :param anchor_coords: numpy array of the anchor coordinates: [x1, y1, x2, y2]
    :param reg_targets: numpy array of the bounding box regression parameters: [tx, ty, tw, th]
    :return: numpy array with the coordinates of the transformed box: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = anchor_coords
    cxa, cya = (x1 + x2) / 2, (y1 + y2) / 2
    wa, ha = x2 - x1, y2 - y1
    tx, ty, tw, th = reg_targets

    cx = tx * wa + cxa
    cy = ty * ha + cya
    w = math.exp(tw) * wa
    h = math.exp(th) * ha
    x = cx - w/2
    y = cy - h/2

    return x, y, x+w, y+h


@profile
def transform_np_all(coords, reg_targets):
    # not used, should this be deleted?
    # coords is x1, y1, x2, y2
    # reg_targets is tx, ty, tw, wh
    x = coords[:, :, :, 0]
    y = coords[:, :, :, 1]
    w = coords[:, :, :, 2] - coords[:, :, :, 0]
    h = coords[:, :, :, 3] - coords[:, :, :, 1]

    tx = reg_targets[:, :, :, 0]
    ty = reg_targets[:, :, :, 1]
    tw = reg_targets[:, :, :, 2]
    th = reg_targets[:, :, :, 3]

    cx = x + w/2
    cy = y + h/2
    cx1 = tx * w + cx
    cy1 = ty * h + cy

    w1 = np.exp(tw.astype(np.float64)) * w
    h1 = np.exp(th.astype(np.float64)) * h
    x1 = cx1 - w1/2
    y1 = cy1 - h1/2

    x1 = np.round(x1)
    y1 = np.round(y1)
    w1 = np.round(w1)
    h1 = np.round(h1)

    return np.stack([x1, y1, x1+w1, y1+h1], axis=3)


@profile
def transform_np_inplace(coords, reg_targets):
    """
    Applies bounding box transformations to multiple boxes. This function is far more efficient than calling transform
    on one box at a time in a loop. This function mutates the input.
    :param coords: 2-d numpy array where each row contains the [x1, y1, x2, y2] coordinates of a box.
    :param reg_targets: 2-numpy array where each row contains the [tx, ty, tw, th] transformation parameters for a box.
    :return: reference to the input array after the transformations have been applied.
    """
    coords[:, 2] -= coords[:, 0]
    coords[:, 3] -= coords[:, 1]

    tx = reg_targets[:, 0]
    ty = reg_targets[:, 1]
    tw = reg_targets[:, 2]
    th = reg_targets[:, 3]

    coords[:,0] += coords[:,2]/2
    coords[:,1] += coords[:,3]/2
    coords[:,0] += tx * coords[:,2]
    coords[:,1] += ty * coords[:,3]

    coords[:,2] *= np.exp(tw)
    coords[:,3] *= np.exp(th)
    coords[:,0] -= coords[:,2]/2
    coords[:,1] -= coords[:,3]/2

    np.round(coords, out=coords)

    coords[:,2] += coords[:,0]
    coords[:,3] += coords[:,1]

    return coords


@profile
def cross_ious(boxes1, boxes2):
    """
    Optimized way of finding all the "intersection over union" overlaps between each box in one set with each box in
    another set. Much faster than calling calc_iou for each individual box pair. This function is optimized for the case
    where boxes2 is smaller than boxes1.
    :param boxes1: 2-d numpy array where each row contains the [x1, y1, x2, y2] coordinates of a box.
    :param boxes2: 2-d numpy array where each row contains the [x1, y1, x2, y2] coordinates of a box.
    :return: 2-d numpy array with dimensions (m x n) where m is the length of boxes1 and n is the length of boxes2. The
     returned array is formatted such that result[i][j] is the IOU between box i of boxes1 and box j of boxes2.
    """
    result = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # assume boxes2 is much smaller than boxes1 so iterate boxes2
    for i, box in enumerate(boxes2):
        x1_intersection = np.maximum(boxes1[:, 0], box[0])
        y1_intersection = np.maximum(boxes1[:, 1], box[1])
        x2_intersection = np.minimum(boxes1[:, 2], box[2])
        y2_intersection = np.minimum(boxes1[:, 3], box[3])

        w_intersection = np.maximum(0, x2_intersection - x1_intersection)
        h_intersection = np.maximum(0, y2_intersection - y1_intersection)

        area_intersection = w_intersection * h_intersection

        area_union = areas1 + areas2[i] - area_intersection

        result[:, i] = (area_intersection / (area_union))

    return result


def get_reg_params(anchor_coords, bbox_coords):
    """
    Finds the bounding box transform parameters needed to transform an anchor into a resulting bounding box.
    :param anchor_coords: list or array containing the anchor coordinates in the format [x1, y1, x2, y2].
    :param bbox_coords: list or array containing the bounding box coordinates in the format [x1, y1, x2, y2].
    :return: tuple of tx, ty, tw, th.
    """
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_coords
    anchor_x1, anchor_y1, anchor_x2, anchor_y2 = anchor_coords

    # convert corners into center and dimensions
    bbox_center_x = (bbox_x2 + bbox_x1) / 2.0
    bbox_center_y = (bbox_y2 + bbox_y1) / 2.0
    bbox_width = bbox_x2 - bbox_x1
    bbox_height = bbox_y2 - bbox_y1

    anchor_center_x = (anchor_x2 + anchor_x1) / 2.0
    anchor_center_y = (anchor_y2 + anchor_y1) / 2.0
    anchor_width = anchor_x2 - anchor_x1
    anchor_height = anchor_y2 - anchor_y1

    tx = (bbox_center_x - anchor_center_x) / anchor_width
    ty = (bbox_center_y - anchor_center_y) / anchor_height
    tw = np.log(bbox_width / anchor_width)
    th = np.log(bbox_height / anchor_height)

    return tx, ty, tw, th


def resize_imgs(imgs, min_size=RESIZE_MIN_SIZE, max_size=RESIZE_MAX_SIZE):
    """
    Resizes images such that the shorter side is min_size pixels, or the longer side is max_size pixels, whichever
    results in a smaller image.
    :param imgs: list of shape.Image objects to resize.
    :param min_size: minimum length in pixels of the shorter side.
    :param max_size: maximum length in pixels of the longer side.
    :return: list of resized images and list of resize ratio corresponding to each image.
    """
    resized_imgs = []
    resized_ratios = []

    for img in imgs:
        resized_img, resized_ratio = img.resize_within_bounds(min_size=min_size, max_size=max_size)
        resized_imgs.append(resized_img)
        resized_ratios.append(resized_ratio)

    return resized_imgs, resized_ratios


def get_bbox_coords(gt_boxes):
    """
    Converts a list of shape.GroundTruthBox objects to a numpy array.
    :param gt_boxes: list of shape.GroundTruthBox objects.
    :return: 2-d numpy array where each row contains the [x1, y1, x2, y2] coordinates of a box.
    """
    bboxes_coords = np.zeros((len(gt_boxes), 4), dtype=np.float32)
    for i, gt_box in enumerate(gt_boxes):
        bboxes_coords[i] = gt_box.corners

    return bboxes_coords


def get_anchors(anchor_scales=DEFAULT_ANCHOR_SCALES, anchor_ratios=DEFAULT_ANCHOR_RATIOS):
    """
    Finds anchor dimensions resulting from a given set of anchor scales and width to height ratios.
    :param anchor_scales: list of integers indicating the square root of the area of the desired anchor in pixels.
    :param anchor_ratios: list of width to height ratios for which each anchor scale should generate anchors.
    :return: list of anchor dimensions.
    """
    naive_anchors = np.array([[size * height, size * width] for size in anchor_scales for height, width in anchor_ratios])
    ratios = np.array([math.sqrt(size * height * size * width) / size for size in anchor_scales for height, width in anchor_ratios])
    anchors = (naive_anchors // ratios[:, None]).astype(int)

    return anchors
