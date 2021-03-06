"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np

DEBUG = False

def bbox_overlaps(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps


def bbox_transform(ex_rois, gt_rois, box_stds):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14) / box_stds[0]
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14) / box_stds[1]
    targets_dw = np.log(gt_widths / ex_widths) / box_stds[2]
    targets_dh = np.log(gt_heights / ex_heights) / box_stds[3]

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bb8_transform(ex_rois, gt_bb8_coordinates, bb8_variance, granularity, im_info):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_bb8_coordinates: [N, 16]
    :param im_info: (H, W, scale)
    :return: [N, 16, granularity[0], granularity[1]]
    """
    assert ex_rois.shape[0] == gt_bb8_coordinates.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    # fine grained anchor centers
    ex_FGA_ctr_x = np.zeros(shape=(ex_rois.shape[0], 1, granularity[1]))
    ex_FGA_ctr_y = np.zeros(shape=(ex_rois.shape[0], granularity[0], 1))
    for i in range(granularity[1]):
        ex_FGA_ctr_x[:, 0, i] = ex_ctr_x + (i - int((granularity[1] - 1) / 2)) * ex_widths / granularity[1]
    ex_FGA_ctr_x = np.repeat(ex_FGA_ctr_x, repeats=granularity[0], axis=1)
    for i in range(granularity[0]):
        ex_FGA_ctr_y[:, i, 0] = ex_ctr_y + (i - int((granularity[0] - 1) / 2)) * ex_heights / granularity[0]
    ex_FGA_ctr_y = np.repeat(ex_FGA_ctr_y, repeats=granularity[1], axis=2)


    # gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    # gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_bb8_coordinates = gt_bb8_coordinates.reshape((gt_bb8_coordinates.shape[0], 8, 2))
    gt_bb8_coordinates_x = gt_bb8_coordinates[:, :, 0] * im_info[1]
    gt_bb8_coordinates_y = gt_bb8_coordinates[:, :, 1] * im_info[0]

    distance_x = gt_bb8_coordinates_x[:, :, np.newaxis, np.newaxis] - ex_FGA_ctr_x[:, np.newaxis]
    distance_y = gt_bb8_coordinates_y[:, :, np.newaxis, np.newaxis] - ex_FGA_ctr_y[:, np.newaxis]
    distance = np.sqrt(np.square(distance_x) + np.square(distance_y))

    FGA_cls_targets = np.zeros_like(distance)
    FGA_cls_targets = FGA_cls_targets.reshape((FGA_cls_targets.shape[0], FGA_cls_targets.shape[1], -1))
    # min_distances = np.min(distance.reshape((distance.shape[0], distance.shape[1], -1)), axis=2)
    index = np.argmin(distance.reshape((distance.shape[0], distance.shape[1], -1)), axis=2)
    for i in range(FGA_cls_targets.shape[0]):
        for j in range(FGA_cls_targets.shape[1]):
            FGA_cls_targets[i, j, index[i, j]] = 1
    FGA_cls_targets = FGA_cls_targets.reshape(distance.shape)

    # shape (N, 8)
    FGA_reg_targets_dx = distance_x[FGA_cls_targets > 0].reshape((distance_x.shape[0], -1))
    FGA_reg_targets_dx = FGA_reg_targets_dx / (ex_widths[:, np.newaxis] / granularity[1] + 1e-14) / bb8_variance[0]
    FGA_reg_targets_dy = distance_y[FGA_cls_targets > 0].reshape((distance_y.shape[0], -1))
    FGA_reg_targets_dy = FGA_reg_targets_dy / (ex_heights[:, np.newaxis] / granularity[0] + 1e-14) / bb8_variance[1]

    # shape (N, 16) xyxy
    FGA_reg_targets = np.stack((FGA_reg_targets_dx, FGA_reg_targets_dy), axis=-1).reshape((FGA_cls_targets.shape[0], -1))
    FGA_reg_weights = np.ones_like(FGA_reg_targets)

    # for softmax target  (N, 8)
    FGA_cls_targets = np.argmax(FGA_cls_targets.reshape(FGA_cls_targets.shape[0], FGA_cls_targets.shape[1], -1), axis=2)

    return FGA_cls_targets, FGA_reg_targets, FGA_reg_weights


def sample_rois(rois, gt_boxes, num_classes, rois_per_image, fg_rois_per_image, fg_overlap, bb8_variance, im_info, granularity):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: [n, 5] (batch_index, x1, y1, x2, y2)
    :param gt_boxes: [n, 40] (cid, x1, y1, x2, y2, difficult, view, inplane, bb8 coordinates (8~24), trans matrix)
    :param num_classes: number of classes
    :param rois_per_image: total roi number
    :param fg_rois_per_image: foreground roi number
    :param fg_overlap: overlap threshold for fg rois
    :param bb8_variance: std var of bb8 reg
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    overlaps = bbox_overlaps(rois[:, 1:], gt_boxes[:, 1:5] * np.array([im_info[1], im_info[0], im_info[1], im_info[0]]))
    gt_assignment = overlaps.argmax(axis=1)
    cid_labels = gt_boxes[gt_assignment, 0]
    max_overlaps = overlaps.max(axis=1)
    if DEBUG:
       print("max_overlaps: {}".format(max_overlaps))

    # select foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(max_overlaps >= fg_overlap)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_this_image = min(fg_rois_per_image, len(fg_indexes))
    # sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_this_image:
        fg_indexes = np.random.choice(fg_indexes, size=fg_rois_this_image, replace=False)

    # select background RoIs as those within [0, FG_THRESH)
    bg_indexes = np.where(max_overlaps < fg_overlap)[0]
    # compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_this_image = rois_per_image - fg_rois_this_image
    bg_rois_this_image = min(bg_rois_this_image, len(bg_indexes))
    # sample bg rois without replacement
    if len(bg_indexes) > bg_rois_this_image:
        bg_indexes = np.random.choice(bg_indexes, size=bg_rois_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)
    # keep_indexes = fg_indexes

    # pad more bg rois to ensure a fixed minibatch size
    while len(keep_indexes) < rois_per_image:
        gap = min(len(bg_indexes), rois_per_image - len(keep_indexes))
        gap_indexes = np.random.choice(range(len(bg_indexes)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, bg_indexes[gap_indexes])

    if DEBUG:
        print("fg_indexes length: {}".format(len(fg_indexes)))
        print("keep_indexes length: {}".format(len(keep_indexes)))
    # sample rois and labels
    rois = rois[keep_indexes]
    # cid_labels = cid_labels[keep_indexes]
    # # set labels of bg rois to be 0
    # cid_labels[fg_rois_this_image:] = 0

    # load or compute bbox_target
    # targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4], box_stds=box_stds)
    FGA_cls_targets, FGA_reg_targets, FGA_reg_weights = \
        bb8_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], 8:24],
                      bb8_variance=bb8_variance, granularity=granularity, im_info=im_info)

    for i in range(fg_rois_this_image, rois_per_image):
        FGA_cls_targets[i] = -1
        FGA_reg_weights[i] = 0

    if DEBUG:
        print("FGA_cls_targets: {}".format(FGA_cls_targets[-1]))
        print("FGA_reg_targets: {}".format(FGA_reg_targets[-1]))
        print("FGA_reg_weights: {}".format(FGA_reg_weights[-1]))

    return rois, FGA_cls_targets, FGA_reg_targets, FGA_reg_weights


class BB8ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_keypoints, batch_images, batch_rois, fg_fraction, fg_overlap, bb8_variance, im_info, granularity):
        super(BB8ProposalTargetOperator, self).__init__()
        self._num_keypoints = num_keypoints
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._rois_per_image = int(batch_rois / batch_images)
        self._fg_rois_per_image = int(round(fg_fraction * self._rois_per_image))
        self._fg_overlap = fg_overlap
        self._bb8_variance = bb8_variance
        self._im_info = im_info
        self._granularity = granularity

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_images == in_data[1].shape[0], 'check batch size of gt_boxes'

        all_rois = in_data[0].asnumpy()    # (batchsize x rpn_post_nms_top_n, 5)
        all_gt_boxes = in_data[1].asnumpy()    # (batchsize, padded label instances, 40)
        if DEBUG:
            print("all_rois shape: {}".format(all_rois.shape))
            print("all_gt_boxes shape: {}".format(all_gt_boxes.shape))

        rois = np.empty((0, 5), dtype=np.float32)
        bb8_FGA_cls_target = np.empty((0, self._num_keypoints), dtype=np.float32)
        bb8_FGA_reg_target = np.empty((0, 2 * self._num_keypoints), dtype=np.float32)
        bb8_FGA_reg_weight = np.empty((0, 2 * self._num_keypoints), dtype=np.float32)
        for batch_idx in range(self._batch_images):
            b_rois = all_rois[np.where(all_rois[:, 0] == batch_idx)[0]]
            b_gt_boxes = all_gt_boxes[batch_idx]
            b_gt_boxes = b_gt_boxes[np.where(b_gt_boxes[:, 0] >= 0)[0]]

            # Include ground-truth boxes in the set of candidate rois
            batch_pad = batch_idx * np.ones((b_gt_boxes.shape[0], 1), dtype=b_gt_boxes.dtype)
            b_rois = np.vstack((b_rois, np.hstack((batch_pad, b_gt_boxes[:, 1:5] * np.array([self._im_info[1], self._im_info[0], self._im_info[1], self._im_info[0]])))))
            if DEBUG:
                print("b_rois shape: {}".format(b_rois.shape))
                print("b_gt_boxes shape: {}".format(b_gt_boxes.shape))
                print("b_rois: {}".format(b_rois[-1]))
                print("b_gt_boxes: {}".format(b_gt_boxes[0]))

            b_rois, b_bb8_FGA_cls_target, b_bb8_FGA_reg_target, b_bb8_FGA_reg_weight = \
                sample_rois(b_rois, b_gt_boxes, num_classes=self._num_keypoints, rois_per_image=self._rois_per_image,
                            fg_rois_per_image=self._fg_rois_per_image, fg_overlap=self._fg_overlap, bb8_variance=self._bb8_variance,
                            im_info=self._im_info, granularity=self._granularity)

            rois = np.vstack((rois, b_rois))
            bb8_FGA_cls_target = np.concatenate((bb8_FGA_cls_target, b_bb8_FGA_cls_target), axis=0)
            bb8_FGA_reg_target = np.vstack((bb8_FGA_reg_target, b_bb8_FGA_reg_target))
            bb8_FGA_reg_weight = np.vstack((bb8_FGA_reg_weight, b_bb8_FGA_reg_weight))

        self.assign(out_data[0], req[0], rois)
        self.assign(out_data[1], req[1], bb8_FGA_cls_target)
        self.assign(out_data[2], req[2], bb8_FGA_reg_target)
        self.assign(out_data[3], req[3], bb8_FGA_reg_weight)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('bb8_proposal_FGAtarget_cls_softmax_reg_offset')
class BB8ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_keypoints='8', batch_images='1', batch_rois='128', fg_fraction='0.25',
                 fg_overlap='0.5', bb8_variance='(0.1, 0.1)', im_info='(512, 512, 1)', granularity='(3, 3)'):
        super(BB8ProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_keypoints = int(num_keypoints)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)
        self._fg_overlap = float(fg_overlap)
        self._bb8_variance = tuple(np.fromstring(bb8_variance[1:-1], dtype=float, sep=','))
        self._im_info = tuple(np.fromstring(im_info[1:-1], dtype=float, sep=','))
        self._granularity = tuple(np.fromstring(granularity[1:-1], dtype=int, sep=','))

    def list_arguments(self):
        return ['rois', 'gt_boxes']

    def list_outputs(self):
        return ['rois_output', 'bb8_FGA_cls_target', 'bb8_FGA_reg_target', 'bb8_FGA_reg_weight']

    def infer_shape(self, in_shape):
        assert self._batch_rois % self._batch_images == 0, \
            'BATCHIMAGES {} must devide BATCH_ROIS {}'.format(self._batch_images, self._batch_rois)

        rpn_rois_shape = in_shape[0]     # (batchsize x rpn_post_nms_top_n, 5)
        gt_boxes_shape = in_shape[1]     # (batchsize, padded_label_instances, 40)

        output_rois_shape = (self._batch_rois, 5)
        bb8_FGA_cls_target_shape = (self._batch_rois, self._num_keypoints)
        # bb8_FGA_cls-agnostic regression
        # bb8_FGA_reg_target_shape = (self._batch_rois, self._num_keypoints * 2)
        # bb8_FGA_reg_weight_shape = (self._batch_rois, self._num_keypoints * 2)
        # bb8_FGA_cls-specific regression, adopt heatmap form
        bb8_FGA_reg_target_shape = (self._batch_rois, self._num_keypoints * 2)
        bb8_FGA_reg_weight_shape = (self._batch_rois, self._num_keypoints * 2)

        return [rpn_rois_shape, gt_boxes_shape], \
               [output_rois_shape, bb8_FGA_cls_target_shape, bb8_FGA_reg_target_shape, bb8_FGA_reg_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return BB8ProposalTargetOperator(self._num_keypoints, self._batch_images, self._batch_rois, self._fg_fraction,
                                      self._fg_overlap, self._bb8_variance, self._im_info, self._granularity)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
