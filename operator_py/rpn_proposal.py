"""
Generate proposals for feature pyramid networks.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool


DEBUG = False

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def clip_boxes_multi_image(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [num_batch, N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [num_batch, N, 4* num_classes]
    """
    # x1 >= 0
    boxes[:, :, 0::4] = np.maximum(np.minimum(boxes[:, :, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, :, 1::4] = np.maximum(np.minimum(boxes[:, :, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, :, 2::4] = np.maximum(np.minimum(boxes[:, :, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, :, 3::4] = np.maximum(np.minimum(boxes[:, :, 3::4], im_shape[0] - 1), 0)
    return boxes

def nonlinear_pred(boxes, box_deltas, variances=(0.1, 0.1, 0.2, 0.2)):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :return: [N, 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * variances[0] * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * variances[1] * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw * variances[2]) * widths[:, np.newaxis]
    pred_h = np.exp(dh * variances[3]) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred_boxes

def nms(dets, thresh, force_suppress=True, num_classes=1):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: NDArray, [[cid, score, x1, y1, x2, y2]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 2]
    y1 = dets[:, 3]
    x2 = dets[:, 4]
    y2 = dets[:, 5]
    scores = dets[:, 1]
    cids = dets[:, 0]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    if force_suppress:
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

    if not force_suppress:
        for cls in range(num_classes):
            indices_cls = np.where(cids[order] == cls)[0]
            order_cls = order[indices_cls]
            while order_cls.size > 0:
                i = order_cls[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order_cls[1:]])
                yy1 = np.maximum(y1[i], y1[order_cls[1:]])
                xx2 = np.minimum(x2[i], x2[order_cls[1:]])
                yy2 = np.minimum(y2[i], y2[order_cls[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order_cls[1:]] - inter)

                inds = np.where(ovr <= thresh)[0]
                order_cls = order_cls[inds + 1]

    return keep

def nonlinear_pred_multi_image(boxes, box_deltas, variances=(0.1, 0.1, 0.2, 0.2)):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [num_batch, N, 4 * num_classes]
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :return: [num_batch, N, 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    pred_boxes = np.zeros(box_deltas.shape)
    for k, box_delta in enumerate(box_deltas):
        dx = box_delta[:, 0::4]
        dy = box_delta[:, 1::4]
        dw = box_delta[:, 2::4]
        dh = box_delta[:, 3::4]

        pred_ctr_x = dx * variances[0] * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * variances[1] * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw * variances[2]) * widths[:, np.newaxis]
        pred_h = np.exp(dh * variances[3]) * heights[:, np.newaxis]

        # pred_boxes = np.zeros(box_delta.shape)
        # x1
        pred_boxes[k, :, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        # y1
        pred_boxes[k, :, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        # x2
        pred_boxes[k, :, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        # y2
        pred_boxes[k, :, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred_boxes

class rpnProposalOperator(mx.operator.CustomOp):
    def __init__(self, output_score,
                 rpn_post_nms_top_n, im_info):
        super(rpnProposalOperator, self).__init__()

        self._im_info = np.fromstring(im_info[1:-1], dtype=int, sep=',')
        # self._num_class = num_class    # excluding background
        self._output_score = output_score
        self._rpn_post_nms_top_n = rpn_post_nms_top_n

        # if DEBUG:
        #     print('feat_stride: {}'.format(self._feat_stride_fpn))
            # print('anchors: {}'.format(self._anchors_fpn))
            # print('num_anchors: {}'.format(self._num_anchors))

    def forward(self, is_train, req, in_data, out_data, aux):
        # rpn_anchor_dict = dict(zip(self.fpn_keys, in_data[2*len(self.fpn_keys):3*len(self.fpn_keys)]))
        rpn_dets = in_data[0].asnumpy()
        # print(rpn_dets.shape)

        batch_size = in_data[0].shape[0]
        # if batch_size > 1:
        #     raise ValueError("Sorry, multiple images each device is not implemented")

        post_nms_topN = self._rpn_post_nms_top_n

        im_info = self._im_info  # in_data[-1].asnumpy()[0, :]

        rois = np.zeros(shape=(batch_size, post_nms_topN, 5))
        cids = np.ones(shape=(batch_size, post_nms_topN, 1)) * -1
        scores = np.zeros(shape=(batch_size, post_nms_topN, 1))
        for k, det in enumerate(rpn_dets):
            keep = np.where(det[:, 0] >= 0)[0]
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]

            if len(keep) > 0:
                if len(keep) < post_nms_topN:
                    # print("keep length: {}".format(len(keep)))
                    pad = npr.choice(keep, size=post_nms_topN - len(keep))
                    keep = np.hstack((keep, pad))
                proposals = det[keep, 2:6] * np.minimum(im_info[0], im_info[1])
                cids[k] = det[keep, 0:1]
                scores[k] = det[keep, 1:2]

                batch_inds = np.ones((proposals.shape[0], 1), dtype=np.float32) * k
                rois[k] = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
            elif len(keep) == 0:
                # print("keep length: {}".format(len(keep)))
                proposals = np.zeros(shape=(post_nms_topN, 4))
                cids[k] = -1
                scores[k] = 0
                batch_inds = np.ones((proposals.shape[0], 1), dtype=np.float32) * k
                rois[k] = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

        self.assign(out_data[0], req[0], rois)

        if self._output_score:
            self.assign(out_data[1], req[1], scores.astype(np.float32, copy=False))
            self.assign(out_data[2], req[2], cids.astype(np.float32, copy=False))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # forward only currently
        self.assign(in_grad[0], req[0], 0)
        # self.assign(in_grad[1], req[1], 0)
        # self.assign(in_grad[2], req[2], 0)


@mx.operator.register("rpn_proposal")
class rpnProposalProp(mx.operator.CustomOpProp):
    def __init__(self, output_score='False',
                 rpn_post_nms_top_n='300', im_info='(512,512,1)'):
        super(rpnProposalProp, self).__init__(need_top_grad=False)

        self._output_score = strtobool(output_score)
        # self._rpn_pre_nms_top_n = int(rpn_pre_nms_top_n)
        self._rpn_post_nms_top_n = int(rpn_post_nms_top_n)
        # self._threshold = float(threshold)
        self._im_info = im_info
        # self._num_class = int(num_class)

    def list_arguments(self):

        return ['rpn_det']

    def list_outputs(self):
        if self._output_score:
            return ['roi', 'score', 'cid']
        else:
            return ['roi']

    def infer_shape(self, in_shape):
        rpn_det_shape = in_shape[0]
        batch_size = rpn_det_shape[0]
        output_shape = (batch_size, self._rpn_post_nms_top_n, 5)
        score_shape = (batch_size, self._rpn_post_nms_top_n, 1)
        cid_shape = (batch_size, self._rpn_post_nms_top_n, 1)

        if self._output_score:
            return in_shape, [output_shape, score_shape, cid_shape]
        else:
            return in_shape, [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return rpnProposalOperator(self._output_score,
                                self._rpn_post_nms_top_n, self._im_info)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
