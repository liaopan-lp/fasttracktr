import numpy as np
import scipy
from scipy.spatial.distance import cdist
import lap

from cython_bbox import bbox_overlaps as bbox_ious
from ..track_utils import kalman_filter


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


# def bbox_ious(atlbrs, btlbrs):
#     """
#     Compute IoU between all boxes from atlbrs with all boxes from btlbrs.
#
#     :param atlbrs: list[tlbr] | np.ndarray, where tlbr is [top-left-x, top-left-y, bottom-right-x, bottom-right-y]
#     :param btlbrs: list[tlbr] | np.ndarray, where tlbr is [top-left-x, top-left-y, bottom-right-x, bottom-right-y]
#     :return: np.ndarray of shape (len(atlbrs), len(btlbrs)) containing IoU values for all combinations
#     """
#     # Convert lists to numpy arrays if they aren't already
#     atlbrs = np.asarray(atlbrs, dtype=np.float32)
#     btlbrs = np.asarray(btlbrs, dtype=np.float32)
#
#     # Get the coordinates of bounding boxes
#     a_x1, a_y1, a_x2, a_y2 = np.split(atlbrs, 4, axis=1)
#     b_x1, b_y1, b_x2, b_y2 = np.split(btlbrs, 4, axis=1)
#
#     # Get the coordinates of the intersection rectangle
#     x1 = np.maximum(a_x1, np.transpose(b_x1))
#     y1 = np.maximum(a_y1, np.transpose(b_y1))
#     x2 = np.minimum(a_x2, np.transpose(b_x2))
#     y2 = np.minimum(a_y2, np.transpose(b_y2))
#
#     # Compute the area of intersection rectangle
#     inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
#
#     # Compute the area of both bounding boxes
#     a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
#     b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
#
#     # Compute IoU
#     iou = inter_area / (a_area + np.transpose(b_area) - inter_area + 1e-7)
#
#     return iou
def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features))  # Nomalized features

    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix
