"""
Generate proposals for feature pyramid networks.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool

# from rcnn.processing.bbox_transform import clip_boxes
# from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
# from rcnn.processing.nms import nms

DEBUG = True

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

class ProposalFPNOperator(mx.operator.CustomOp):
    def __init__(self, feat_stride_fpn, scales, ratios, output_score, num_class,
                 rpn_pre_nms_top_n, rpn_post_nms_top_n, threshold, rpn_min_size_fpn, im_info):
        super(ProposalFPNOperator, self).__init__()
        self._feat_stride_fpn = np.fromstring(feat_stride_fpn[1:-1], dtype=int, sep=',')
        self.fpn_keys = []
        fpn_stride = []

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s'%s)
            fpn_stride.append(int(s))

        self._scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self._ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')
        self._im_info = np.fromstring(im_info[1:-1], dtype=int, sep=',')
        # self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(base_size=fpn_stride, scales=self._scales, ratios=self._ratios)))
        # self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        self._num_anchors = dict(zip(self.fpn_keys, [4,4,4,4,4]))
        self._num_class = num_class    # excluding background
        self._output_score = output_score
        self._rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self._rpn_post_nms_top_n = rpn_post_nms_top_n
        self._threshold = threshold
        self._rpn_min_size_fpn = dict(zip(self.fpn_keys, np.fromstring(rpn_min_size_fpn[1:-1], dtype=int, sep=',')))
        self._bbox_pred = nonlinear_pred

        if DEBUG:
            print('feat_stride: {}'.format(self._feat_stride_fpn))
            # print('anchors: {}'.format(self._anchors_fpn))
            # print('num_anchors: {}'.format(self._num_anchors))

    def forward(self, is_train, req, in_data, out_data, aux):
        # nms = gpu_nms_wrapper(self._threshold, in_data[0][0].context.device_id)

        cls_prob_dict = dict(zip(self.fpn_keys, in_data[0:len(self.fpn_keys)]))
        bbox_pred_dict = dict(zip(self.fpn_keys, in_data[len(self.fpn_keys):2*len(self.fpn_keys)]))
        rpn_anchor_dict = dict(zip(self.fpn_keys, in_data[2*len(self.fpn_keys):3*len(self.fpn_keys)]))

        batch_size = in_data[0].shape[0]
        if batch_size > 1:
            raise ValueError("Sorry, multiple images each device is not implemented")

        pre_nms_topN = self._rpn_pre_nms_top_n
        post_nms_topN = self._rpn_post_nms_top_n
        min_size_dict = self._rpn_min_size_fpn
        im_info = self._im_info  # in_data[-1].asnumpy()[0, :]

        proposals_list = []
        cids_list = []
        scores_list = []
        for s in self._feat_stride_fpn:
            stride = int(s)

            height, width = int(im_info[0] / stride), int(im_info[1] / stride)
            A = self._num_anchors['stride%s' % s]
            K = height * width

            scores = cls_prob_dict['stride%s' % s].asnumpy()  # [:, self._num_anchors['stride%s'%s]:, :, :]
            bbox_deltas = bbox_pred_dict['stride%s' % s].asnumpy()
            # anchors = anchors_plane(height, width, stride, self._anchors_fpn['stride%s'%s].astype(np.float32))
            anchors = rpn_anchor_dict['stride%s' % s].asnumpy()

            if DEBUG:
                print("stride: {}".format(stride))
                print("shape of cls_prob: {}".format(scores.shape))
                print("cls_prob: {}".format(scores[0, 0:9, 0, 0]))
                print("shape of bbox_pred: {}".format(bbox_deltas.shape))
                print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
                print('scale: {}'.format(im_info[2]))
                print("feature map height and width: {}, {}".format(height, width))
                print("shape of anchor: {}".format(anchors.shape))

            anchors = anchors.reshape((K * A, 4)) * np.minimum(im_info[0], im_info[1])

            bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((K * A, 4))

            scores = self._clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((K * A, -1))    # (K * A, num_class)

            cid = np.argmax(scores, axis=-1) - 1    # background -1, foreground 0~7
            cid = cid.reshape((K * A, 1))
            scores = np.max(scores[:, 1:], axis=-1, keepdims=True)
            if DEBUG:
                print("cid[0]: {}".format(cid[0]))
                print("scores[0]: {}".format(scores[0]))

            proposals = self._bbox_pred(anchors, bbox_deltas)    #TODO:
            if DEBUG:
                print('bbox_delta: {}'.format(bbox_deltas[0,:]))
                print('anchor: {}'.format(anchors[0, :]))
                print('proposals: {}'.format(proposals[0, :]))
                print("shape of proposals: {}".format(proposals.shape))

            proposals = clip_boxes(proposals, im_info[:2])

            keep = self._filter_boxes(proposals, min_size_dict['stride%s'%s] * im_info[2])
            proposals = proposals[keep, :]
            cid = cid[keep]
            scores = scores[keep]
            if DEBUG:
                print("shape of filtered proposals: {}".format(proposals.shape))
            proposals_list.append(proposals)
            cids_list.append(cid)
            scores_list.append(scores)

        proposals = np.vstack(proposals_list)    #TODO
        cids = np.vstack(cids_list)
        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        #np.ravel()
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        cids = cids[order]
        scores = scores[order]

        det = np.hstack((cids, scores, proposals)).astype(np.float32)

        if np.shape(det)[0] == 0:
            print("Something wrong with the input image(resolution is too low?), generate fake proposals for it.")
            proposals = np.array([[1.0, 1.0, 2.0, 2.0]]*post_nms_topN, dtype=np.float32)
            cids = np.array([[0.0]] * post_nms_topN, dtype=np.float32)
            scores = np.array([[0.9]]*post_nms_topN, dtype=np.float32)
            det = np.array([[0.0, 0.9, 1.0, 1.0, 2.0, 2.0]]*post_nms_topN, dtype=np.float32)

        keep = nms(det, thresh=self._threshold, force_suppress=False, num_classes=self._num_class)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]

        if len(keep) < post_nms_topN:
            pad = npr.choice(keep, size=post_nms_topN - len(keep))
            keep = np.hstack((keep, pad))
        proposals = proposals[keep, :]
        cids = cids[keep]
        scores = scores[keep]

        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        self.assign(out_data[0], req[0], blob)

        if self._output_score:
            self.assign(out_data[1], req[1], scores.astype(np.float32, copy=False))
            self.assign(out_data[2], req[2], cids.astype(np.float32, copy=False))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # forward only currently
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
        Clip boxes of the pad area.
        :param tensor: [n, c, H, W]
        :param pad_shape: [h, w]
        :return: [n, c, h, w]
        """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor


@mx.operator.register("proposal_fpn")
class ProposalFPNProp(mx.operator.CustomOpProp):
    def __init__(self, feat_stride='(64,32,16,8,4)', scales='(8)', ratios='(0.5, 1, 2)', output_score='False', num_class='1',
                 rpn_pre_nms_top_n='6000', rpn_post_nms_top_n='300', threshold='0.3', rpn_min_size='(64,32,16,8,4)', im_info='(512,512,1)'):
        super(ProposalFPNProp, self).__init__(need_top_grad=False)
        self._feat_stride_fpn = feat_stride
        self._scales = scales
        self._ratios = ratios
        self._output_score = strtobool(output_score)
        self._rpn_pre_nms_top_n = int(rpn_pre_nms_top_n)
        self._rpn_post_nms_top_n = int(rpn_post_nms_top_n)
        self._threshold = float(threshold)
        self._rpn_min_size_fpn = rpn_min_size
        self._im_info = im_info
        self._num_class = int(num_class)

    def list_arguments(self):
        args_list = []
        for s in np.fromstring(self._feat_stride_fpn[1:-1], dtype=int, sep=','):
            args_list.append('cls_prob_stride%s' % s)
        for s in np.fromstring(self._feat_stride_fpn[1:-1], dtype=int, sep=','):
            args_list.append('bbox_pred_stride%s' % s)
        for s in np.fromstring(self._feat_stride_fpn[1:-1], dtype=int, sep=','):
            args_list.append('rpn_anchor_stride%s' % s)
        # args_list.append('im_info')

        return args_list

    def list_outputs(self):
        if self._output_score:
            return ['output', 'score', 'cid']
        else:
            return ['output']

    def infer_shape(self, in_shape):
        output_shape = (self._rpn_post_nms_top_n, 5)
        score_shape = (self._rpn_post_nms_top_n, 1)
        cid_shape = (self._rpn_post_nms_top_n, 1)

        if self._output_score:
            return in_shape, [output_shape, score_shape, cid_shape]
        else:
            return in_shape, [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalFPNOperator(self._feat_stride_fpn, self._scales, self._ratios, self._output_score, self._num_class,
                                self._rpn_pre_nms_top_n, self._rpn_post_nms_top_n, self._threshold, self._rpn_min_size_fpn, self._im_info)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
