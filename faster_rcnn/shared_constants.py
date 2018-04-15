import math

import numpy as np

BBREG_MULTIPLIERS = np.array([10, 10, 5, 5], dtype=np.float32)

DEFAULT_ANCHOR_SCALES = np.array([16, 32, 64, 128, 256, 512])
DEFAULT_ANCHOR_RATIOS = np.array([[1, 1], [1, 2], [2, 1]])
_NAIVE_ANCHORS = np.array([[size * height, size * width] for size in DEFAULT_ANCHOR_SCALES for height, width in DEFAULT_ANCHOR_RATIOS])
_RATIOS = np.array([math.sqrt(size * height * size * width) / size for size in DEFAULT_ANCHOR_SCALES for height, width in DEFAULT_ANCHOR_RATIOS])
DEFAULT_ANCHORS = (_NAIVE_ANCHORS // _RATIOS[:, None]).astype(int)
DEFAULT_ANCHORS_PER_LOC = len(DEFAULT_ANCHORS)
DEFAULT_NUM_ITERATIONS = 10
DEFAULT_LEARN_RATE = 1e-3
DEFAULT_MOMENTUM = 0.9
RESIZE_MIN_SIZE = 600
RESIZE_MAX_SIZE = 1000
NUM_ROIS = 64