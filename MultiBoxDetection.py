import mxnet as mx
import numpy as np

def CalculateOverlap(a, b):
    """
    Calculate intersection-over-union overlap
    Params:
    ----------
    a : NDArray
        single box [xmin, ymin ,xmax, ymax]
    b : NDArray
        single box [xmin, ymin, xmax, ymax]
    Returns:
    -----------
    """
    w = mx.nd.maximum(0, mx.nd.minimum(a[2], b[2]) - mx.nd.maximum(a[0], b[0]))
    h = mx.nd.maximum(0, mx.nd.minimum(a[3], b[3]) - mx.nd.maximum(a[1], b[1]))
    i = w * h
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    iou = i / u if u > 0 else 0
    return iou


def TransformLocations(anchors=None, loc_pred=None, clip=True, variances=(0.1, 0.1, 0.2, 0.2)):
    """
    :param anchors: NDArray, 1 x num_anchors x 4
    :param loc_pred: NDArray, batchsize x (num_anchors x 4)
    :param clip: (boolean, optional, default=True) clip out-of-boundary boxes
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :return: output: NDArray, batchsize x num_anchors x 4, locations in [xmin, ymin, xmax, ymax]
    """
    loc_pred = loc_pred.reshape((0, -1, 4))

    al = anchors[:, :, 0:1]
    at = anchors[:, :, 1:2]
    ar = anchors[:, :, 2:3]
    ab = anchors[:, :, 3:4]
    aw = ar - al
    ah = ab - at
    acx = (al + ar) / 2.0
    acy = (at + ab) / 2.0

    px = loc_pred[:, :, 0:1]
    py = loc_pred[:, :, 1:2]
    pw = loc_pred[:, :, 2:3]
    ph = loc_pred[:, :, 3:4]

    ox = px * variances[0] * aw + acx
    oy = py * variances[1] * ah + acy
    ow = mx.nd.exp(pw * variances[2]) * aw / 2.0
    oh = mx.nd.exp(ph * variances[3]) * ah / 2.0

    out = mx.nd.zeros_like(loc_pred)
    if not clip:
        out = mx.nd.concat(ox-ow, oy-oh, ox+ow, oy+oh, dim=2)
    if clip:
        column0 = mx.nd.maximum(0, mx.nd.minimum(1, ox-ow))
        column1 = mx.nd.maximum(0, mx.nd.minimum(1, oy-oh))
        column2 = mx.nd.maximum(0, mx.nd.minimum(1, ox+ow))
        column3 = mx.nd.maximum(0, mx.nd.minimum(1, oy+oh))
        out = mx.nd.concat(column0, column1, column2, column3, dim=2)

    return out


def TransformBB8(anchors=None, bb8_pred=None, clip=True, variances=(0.1, 0.1, 0.2, 0.2)):
    """
    :param anchors: NDArray, 1 x num_anchors x 4
    :param bb8_pred: NDArray, batchsize x (num_anchors x 16)
    :param clip: (boolean, optional, default=True) clip out-of-boundary boxes
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :return: output: NDArray, batchsize x num_anchors x 16, locations in [x, y, x, y, ...]
    """
    bb8_pred = mx.nd.reshape(bb8_pred, shape=(0, -1, 16))

    al = anchors[:, :, 0:1]
    at = anchors[:, :, 1:2]
    ar = anchors[:, :, 2:3]
    ab = anchors[:, :, 3:4]
    aw = ar - al
    ah = ab - at
    acx = (al + ar) / 2.0
    acy = (at + ab) / 2.0

    anchor_wh = mx.nd.concat(aw, ah, dim=2)
    anchor_wh = mx.nd.tile(anchor_wh, reps=(1,1,8))
    anchor_center = mx.nd.concat(acx, acy, dim=2)
    anchor_center = mx.nd.tile(anchor_center, reps=(1,1,8))

    out = bb8_pred * anchor_wh * variances[0] + anchor_center

    if clip:
        out = mx.nd.maximum(0, mx.nd.minimum(1, out))

    return out


def TransformRCNNFGABB8(rois_concat, rcnn_FGA_cls_score_concat, rcnn_FGA_bb8_pred_concat,
                        clip=True, variances=(0.1, 0.1, 0.2, 0.2)):
    """
    :param rois_concat: batchsize x num_anchors x 4
    :param rcnn_FGA_cls_score_concat: batchsize x num_anchors x num_keypoints x granularity[0] x granularity[1]
    :param rcnn_FGA_bb8_pred_concat: batchsize x num_anchors x (2x num_keypoints) x granularity[0] x granularity[1]
    :param clip: (boolean, optional, default=True) clip out-of-boundary boxes
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :return: output: batchsize x num_anchors x 16, locations in [x, y, x, y, ...]
    """
    batchsize = rois_concat.shape[0]
    num_anchors = rois_concat.shape[1]
    num_keypoints = rcnn_FGA_cls_score_concat.shape[2]
    granularity = tuple([rcnn_FGA_cls_score_concat.shape[3], rcnn_FGA_cls_score_concat.shape[4]])
    # bb8_pred = mx.nd.reshape(bb8_pred, shape=(0, -1, 16))

    al = rois_concat[:, :, 0:1]
    at = rois_concat[:, :, 1:2]
    ar = rois_concat[:, :, 2:3]
    ab = rois_concat[:, :, 3:4]
    aw = ar - al
    ah = ab - at
    acx = (al + ar) / 2.0
    acy = (at + ab) / 2.0

    # fine grained anchor centers
    ex_FGA_ctr_x = np.zeros(shape=(batchsize, num_anchors, 1, granularity[1]))
    ex_FGA_ctr_y = np.zeros(shape=(batchsize, num_anchors, granularity[0], 1))
    for i in range(granularity[1]):
        ex_FGA_ctr_x[:, :, 0, i:i+1] = acx + (i - int((granularity[1] - 1) / 2)) * aw / granularity[1]
    ex_FGA_ctr_x = np.repeat(ex_FGA_ctr_x, repeats=granularity[0], axis=2)
    for i in range(granularity[0]):
        ex_FGA_ctr_y[:, :, i:i+1, 0] = acy + (i - int((granularity[0] - 1) / 2)) * ah / granularity[0]
    ex_FGA_ctr_y = np.repeat(ex_FGA_ctr_y, repeats=granularity[1], axis=3)

    ex_FGA_ctr_x = np.reshape(ex_FGA_ctr_x, newshape=(batchsize, num_anchors, -1))
    ex_FGA_ctr_x = np.repeat(ex_FGA_ctr_x[:, :, np.newaxis], repeats=num_keypoints, axis=2)
    ex_FGA_ctr_y = np.reshape(ex_FGA_ctr_y, newshape=(batchsize, num_anchors, -1))
    ex_FGA_ctr_y = np.repeat(ex_FGA_ctr_y[:, :, np.newaxis], repeats=num_keypoints, axis=2)

    rcnn_FGA_cls_score_concat = rcnn_FGA_cls_score_concat.reshape((batchsize, num_anchors, num_keypoints, -1))
    FGA_id = np.argmax(rcnn_FGA_cls_score_concat, axis=-1)
    rcnn_FGA_bb8_pred_concat = rcnn_FGA_bb8_pred_concat.reshape((batchsize, num_anchors, 2 * num_keypoints, -1))
    # rcnn_FGA_bb8_pred_concat = rcnn_FGA_bb8_pred_concat.reshape((batchsize, num_anchors, num_keypoints, 2))
    index_0 = np.repeat(np.arange(0, batchsize), repeats=num_anchors * num_keypoints)
    index_0 = np.reshape(index_0, newshape=(batchsize, num_anchors, num_keypoints))
    index_1 = np.repeat(np.arange(0, num_anchors), repeats=num_keypoints)
    index_1 = np.tile(index_1, reps=batchsize)
    index_1 = np.reshape(index_1, newshape=(batchsize, num_anchors, num_keypoints))
    index_2x = np.tile(np.arange(0, num_keypoints), reps=batchsize * num_anchors)
    index_2y = np.tile(np.arange(num_keypoints, 2 * num_keypoints), reps=batchsize * num_anchors)
    index_2x = np.reshape(index_2x, newshape=(batchsize, num_anchors, num_keypoints))
    index_2y = np.reshape(index_2y, newshape=(batchsize, num_anchors, num_keypoints))

    rcnn_FGA_bb8_delta_x_pick = rcnn_FGA_bb8_pred_concat[index_0, index_1, index_2x, FGA_id]
    rcnn_FGA_bb8_delta_y_pick = rcnn_FGA_bb8_pred_concat[index_0, index_1, index_2y, FGA_id]
    # rcnn_FGA_bb8_delta_x_pick = rcnn_FGA_bb8_pred_concat[:, :, :, 0]
    # rcnn_FGA_bb8_delta_y_pick = rcnn_FGA_bb8_pred_concat[:, :, :, 1]
    ex_FGA_ctr_x_pick = ex_FGA_ctr_x[index_0, index_1, index_2x, FGA_id]
    ex_FGA_ctr_y_pick = ex_FGA_ctr_y[index_0, index_1, index_2x, FGA_id]

    rcnn_FGA_bb8_pred_x = rcnn_FGA_bb8_delta_x_pick * aw * variances[0] + ex_FGA_ctr_x_pick
    rcnn_FGA_bb8_pred_y = rcnn_FGA_bb8_delta_y_pick * ah * variances[1] + ex_FGA_ctr_y_pick

    out = np.concatenate((rcnn_FGA_bb8_pred_x[:, :, :, np.newaxis], rcnn_FGA_bb8_pred_y[:, :, :, np.newaxis]),
                         axis=-1)
    out = np.reshape(out, newshape=(batchsize, num_anchors, 2 * num_keypoints))   # the last dimension is [x,y,x,y,...]

    if clip:
        out = np.maximum(0, np.minimum(1, out))

    return out

def TransformMaskRCNNKeypointBB8(rois_concat, maskrcnn_keypoint_cls_score_concat,
                        clip=True):
    """
    :param rois_concat: batchsize x num_anchors x 4, float number in range [0, 1]
    :param maskrcnn_keypoint_cls_score_concat: batchsize x num_anchors x num_keypoints x granularity[0] x granularity[1]
    :param clip: (boolean, optional, default=True) clip out-of-boundary boxes
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :return: output: batchsize x num_anchors x 16, locations in [x, y, x, y, ...]
    """
    batchsize = rois_concat.shape[0]
    num_anchors = rois_concat.shape[1]
    num_keypoints = maskrcnn_keypoint_cls_score_concat.shape[2]
    granularity = tuple([maskrcnn_keypoint_cls_score_concat.shape[3], maskrcnn_keypoint_cls_score_concat.shape[4]])

    al = rois_concat[:, :, 0:1]
    at = rois_concat[:, :, 1:2]
    ar = rois_concat[:, :, 2:3]
    ab = rois_concat[:, :, 3:4]
    aw = ar - al
    ah = ab - at

    maskrcnn_keypoint_cls_score_concat = maskrcnn_keypoint_cls_score_concat.reshape((batchsize, num_anchors, num_keypoints, -1))

    keypoints_label = np.argmax(maskrcnn_keypoint_cls_score_concat, axis=-1)
    keypoint_score = np.max(maskrcnn_keypoint_cls_score_concat, axis=-1)

    J_y = keypoints_label // granularity[1] # shape [batchsize, num_anchors, num_keypoints]
    J_x = keypoints_label % granularity[1]

    x_scale = aw / granularity[1]
    y_scale = ah / granularity[0]
    x_shift = al
    y_shift = at
    J_x = np.array(x_scale * (J_x + 0.5)) + x_shift
    J_y = np.array(y_scale * (J_y + 0.5)) + y_shift

    out = np.stack((J_x, J_y), axis=-1)
    out = np.reshape(out, newshape=(batchsize, num_anchors, 2 * num_keypoints))   # the last dimension is [x,y,x,y,...]

    if clip:
        out = np.maximum(0, np.minimum(1, out))

    return out

def TransformRCNNOffsetBB8(rois_concat, rcnn_bb8offset_pred_concat,
                        clip=True, bb8variances=(0.1, 0.1)):
    """
    :param rois_concat: batchsize x num_anchors x 4
    :param rcnn_bb8offset_pred_concat: batchsize x num_anchors x (num_keypoints) x 2
    :param clip: (boolean, optional, default=True) clip out-of-boundary boxes
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :return: output: batchsize x num_anchors x 16, locations in [x, y, x, y, ...]
    """
    batchsize = rois_concat.shape[0]
    num_anchors = rois_concat.shape[1]
    num_keypoints = rcnn_bb8offset_pred_concat.shape[2]

    al = rois_concat[:, :, 0:1]
    at = rois_concat[:, :, 1:2]
    ar = rois_concat[:, :, 2:3]
    ab = rois_concat[:, :, 3:4]
    aw = ar - al
    ah = ab - at
    # shape [batchsize, num_anchors, 1]
    acx = (al + ar) / 2.0
    acy = (at + ab) / 2.0

    rcnn_bb8offset_pred_x = rcnn_bb8offset_pred_concat[:, :, :, 0] * aw * bb8variances[0] + acx
    rcnn_bb8offset_pred_y = rcnn_bb8offset_pred_concat[:, :, :, 1] * ah * bb8variances[1] + acy

    out = np.concatenate((rcnn_bb8offset_pred_x[:, :, :, np.newaxis], rcnn_bb8offset_pred_y[:, :, :, np.newaxis]),
                         axis=-1)
    out = np.reshape(out, newshape=(batchsize, num_anchors, 2 * num_keypoints))   # the last dimension is [x,y,x,y,...]

    if clip:
        out = np.maximum(0, np.minimum(1, out))

    return out


def nms(dets, thresh, force_suppress=True, num_classes=1):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: NDArray, [[cid, score, x1, y1, x2, y2]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 2].asnumpy()
    y1 = dets[:, 3].asnumpy()
    x2 = dets[:, 4].asnumpy()
    y2 = dets[:, 5].asnumpy()
    scores = dets[:, 1].asnumpy()
    cids = dets[:, 0].asnumpy()

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


def myMultiBoxDetection(cls_prob, loc_pred, anchors, \
                    threshold=0.01, clip=True, background_id=0, nms_threshold=0.45, force_suppress=False,
                    variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400, name=None):
    """
    Parameters:
    :param cls_prob: class probabilities, batchsize x (num_classes + 1) x num_anchors
    :param loc_pred: location regression predictions, batchsize x (num_anchors x 4)
    :param anchors: multibox prior anchor boxes, 1 x num_anchors x 4
    :param threshold: (float, optional, default=0.01) threshold to be a positive prediction
    :param clip: (boolean, optional, default=True) clip out-of-boundary boxes
    :param background_id: (int optional, default='0') background id
    :param nms_threshold: (float, optional, default=0.5) non-maximum suppression threshold
    :param force_suppress: (boolean, optional, default=False) suppress all detections regardless of class_id
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :param nms_topk: (int, optional, default=-1) keep maximum top k detections before nms, -1 for no limit.
    :param out: (NDArray, optional) the output NDArray to hold the result.
    :param name:

    :return: out: (NDArray or list of NDArray) the output of this function.
    """
    assert background_id == 0, "No implementation for background_id is not 0!!"
    assert len(variances) == 4, "Variance size must be 4"
    assert nms_threshold > 0, "NMS_threshold should be greater than 0!!!"
    assert nms_threshold <=1, "NMS_threshold should be less than 1!!!"

    # ctx = cls_prob.context
    num_classes = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]
    num_batches = cls_prob.shape[0]

    out = mx.nd.ones(shape=(num_batches, num_anchors, 6)) * -1
    # remove background, restore original id
    out[:, :, 0] = mx.nd.argmax(cls_prob[:, 1:, :], axis=1, keepdims=False)
    out[:, :, 1] = mx.nd.max(cls_prob[:, 1:, :], axis=1, keepdims=False, exclude=False)
    out[:, :, 2:6] = TransformLocations(anchors, loc_pred, clip, variances)

    # if the score < positive threshold, reset the id and score to -1
    out[:, :, 0] = mx.nd.where(condition=out[:, :, 1]<threshold,
                x=mx.nd.ones_like(out[:, :, 1]) * -1,
                y=out[:, :, 0])
    out[:, :, 1] = mx.nd.where(condition=out[:, :, 1] < threshold,
                               x=mx.nd.ones_like(out[:, :, 1]) * -1,
                               y=out[:, :, 1])

    valid_count = mx.nd.sum(out[:, :, 0] >= 0, axis=0, keepdims=False, exclude=True)

    #*******************************************************************************************

    for nbatch in range(num_batches):
        p_out = out[nbatch, :, :]

        if (valid_count[nbatch] < 1) or (nms_threshold <= 0) or (nms_threshold > 1):
            continue

        # sort and apply NMS
        nkeep = nms_topk if nms_topk<valid_count[nbatch] else valid_count[nbatch]
        # sort confidence in descend order and re-order output
        p_out[0:nkeep] = p_out[p_out[:, 1].topk(k=nkeep)]
        p_out[nkeep:, 0] = -1    # not performed in original mxnet MultiBoxDetection, add by zhangxin

        # apply nms
        keep_indices = nms(p_out[0:nkeep], nms_threshold, force_suppress, num_classes-1)
        keep_indices = np.array(keep_indices)
        p_out[0:len(keep_indices)] = p_out[keep_indices]
        p_out[len(keep_indices):, 0] = -1
        out[nbatch, :, :] = p_out

    return out



def BB8MultiBoxDetection(cls_prob, loc_pred, bb8_pred, anchors, \
                    threshold=0.01, clip=True, background_id=0, nms_threshold=0.45, force_suppress=False,
                    variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400, name=None):
    """
    Parameters:
    :param cls_prob: class probabilities, batchsize x (num_classes + 1) x num_anchors
    :param loc_pred: location regression predictions, batchsize x (num_anchors x 4)
    :param bb8_pred: bb8 regression predictions, batchsize x (num_anchors x 16)
    :param anchors: multibox prior anchor boxes, 1 x num_anchors x 4
    :param threshold: (float, optional, default=0.01) threshold to be a positive prediction
    :param clip: (boolean, optional, default=True) clip out-of-boundary boxes
    :param background_id: (int optional, default='0') background id
    :param nms_threshold: (float, optional, default=0.5) non-maximum suppression threshold
    :param force_suppress: (boolean, optional, default=False) suppress all detections regardless of class_id
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :param nms_topk: (int, optional, default=-1) keep maximum top k detections before nms, -1 for no limit.
    :param out: (NDArray, optional) the output NDArray to hold the result.
    :param name:

    :return: out: (NDArray or list of NDArray) the output of this function.
    """
    assert background_id == 0, "No implementation for background_id is not 0!!"
    assert len(variances) == 4, "Variance size must be 4"
    assert nms_threshold > 0, "NMS_threshold should be greater than 0!!!"
    assert nms_threshold <=1, "NMS_threshold should be less than 1!!!"

    # ctx = cls_prob.context
    num_classes = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]
    num_batches = cls_prob.shape[0]

    out = mx.nd.ones(shape=(num_batches, num_anchors, 22)) * -1
    # remove background, restore original id
    out[:, :, 0] = mx.nd.argmax(cls_prob[:, 1:, :], axis=1, keepdims=False)
    out[:, :, 1] = mx.nd.max(cls_prob[:, 1:, :], axis=1, keepdims=False, exclude=False)
    out[:, :, 2:6] = TransformLocations(anchors, loc_pred, clip, variances)
    out[:, :, 6:22] = TransformBB8(anchors, bb8_pred, clip, variances)

    # if the score < positive threshold, reset the id and score to -1
    out[:, :, 0] = mx.nd.where(condition=out[:, :, 1]<threshold,
                x=mx.nd.ones_like(out[:, :, 1]) * -1,
                y=out[:, :, 0])
    out[:, :, 1] = mx.nd.where(condition=out[:, :, 1] < threshold,
                               x=mx.nd.ones_like(out[:, :, 1]) * -1,
                               y=out[:, :, 1])

    valid_count = mx.nd.sum(out[:, :, 0] >= 0, axis=0, keepdims=False, exclude=True)
    valid_count = valid_count.asnumpy()

    #*******************************************************************************************

    for nbatch in range(num_batches):
        p_out = out[nbatch, :, :]

        if (valid_count[nbatch] < 1) or (nms_threshold <= 0) or (nms_threshold > 1):
            continue

        # sort and apply NMS
        nkeep = nms_topk if nms_topk<valid_count[nbatch] else int(valid_count[nbatch])
        # sort confidence in descend order and re-order output
        p_out[0:nkeep] = p_out[p_out[:, 1].topk(k=nkeep)]
        p_out[nkeep:, 0] = -1    # not performed in original mxnet MultiBoxDetection, add by zhangxin

        # apply nms
        keep_indices = nms(p_out[0:nkeep], nms_threshold, force_suppress, num_classes-1)
        keep_indices = np.array(keep_indices)
        p_out[0:len(keep_indices)] = p_out[keep_indices]
        p_out[len(keep_indices):, 0] = -1
        out[nbatch, :, :] = p_out

    return out


def RCNNFGABB8MultiBoxDetection(rois_concat, score_concat, cid_concat, rcnn_FGA_cls_score_concat,
                                       rcnn_FGA_bb8_pred_concat,
                    threshold=0.01, clip=True, background_id=0, nms_threshold=0.45, force_suppress=False,
                    variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400, name=None, im_info=(512,512,1)):
    """
    Parameters:
    :param rois_concat: batchsize x (num_post_nms x num_fpn) x 5
    :param score_concat: batchsize x (num_post_nms x num_fpn) x 1
    :param cid_concat: batchsize x (num_post_nms x num_fpn) x 1
    :param rcnn_FGA_cls_score_concat: (batchsize x num_post_nms x num_fpn) x num_keypoints x granularity[0] x granularity[1]
    :param rcnn_FGA_bb8_pred_concat: (batchsize x num_post_nms x num_fpn) x (2 x num_keypoints) x granularity[0] x granularity[1]
    :param threshold: (float, optional, default=0.01) threshold to be a positive prediction
    :param clip: (boolean, optional, default=True) clip out-of-boundary boxes
    :param background_id: (int optional, default='0') background id
    :param nms_threshold: (float, optional, default=0.5) non-maximum suppression threshold
    :param force_suppress: (boolean, optional, default=False) suppress all detections regardless of class_id
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :param nms_topk: (int, optional, default=-1) keep maximum top k detections before nms, -1 for no limit.
    :param out: (NDArray, optional) the output NDArray to hold the result.
    :param name:

    :return: out: (NDArray or list of NDArray) the output of this function.
    """
    assert background_id == 0, "No implementation for background_id is not 0!!"
    assert len(variances) == 4, "Variance size must be 4"
    assert nms_threshold > 0, "NMS_threshold should be greater than 0!!!"
    assert nms_threshold <=1, "NMS_threshold should be less than 1!!!"

    # ctx = cls_prob.context
    num_batches = score_concat.shape[0]
    num_rois_per_image = score_concat.shape[1]
    num_keypoints = rcnn_FGA_cls_score_concat.shape[1]
    num_classes = int(np.max(cid_concat) + 1)
    granularity = tuple([rcnn_FGA_cls_score_concat.shape[2], rcnn_FGA_cls_score_concat.shape[3]])
    rois_concat[:, 1:5] /= np.array([im_info[1], im_info[0], im_info[1], im_info[0]])
    rois_concat = rois_concat.reshape((num_batches, num_rois_per_image, 5))
    rcnn_FGA_cls_score_concat = np.reshape(rcnn_FGA_cls_score_concat,
                                           newshape=(num_batches, num_rois_per_image, num_keypoints,
                                                    granularity[0],
                                                    granularity[1]))
    rcnn_FGA_bb8_pred_concat = np.reshape(rcnn_FGA_bb8_pred_concat,
                                           newshape=(num_batches, num_rois_per_image, 2 *num_keypoints,
                                                     granularity[0],
                                                     granularity[1]))
    # rcnn_FGA_bb8_pred_concat = np.reshape(rcnn_FGA_bb8_pred_concat,
    #                                       newshape=(num_batches, num_rois_per_image, 2 * num_keypoints))

    out = np.ones(shape=(num_batches, num_rois_per_image, 22)) * -1
    # remove background, restore original id
    out[:, :, 0:1] = cid_concat
    out[:, :, 1:2] = score_concat
    out[:, :, 2:6] = rois_concat[:, :, 1:5]
    out[:, :, 6:22] = TransformRCNNFGABB8(rois_concat[:, :, 1:5], rcnn_FGA_cls_score_concat, rcnn_FGA_bb8_pred_concat, clip, variances)
    out = mx.nd.array(out)

    # if the score < positive threshold, reset the id and score to -1
    out[:, :, 0] = mx.nd.where(condition=out[:, :, 1]<threshold,
                x=mx.nd.ones_like(out[:, :, 1]) * -1,
                y=out[:, :, 0])
    out[:, :, 1] = mx.nd.where(condition=out[:, :, 1] < threshold,
                               x=mx.nd.ones_like(out[:, :, 1]) * -1,
                               y=out[:, :, 1])

    valid_count = mx.nd.sum(out[:, :, 0] >= 0, axis=0, keepdims=False, exclude=True)
    valid_count = valid_count.asnumpy()

    #*******************************************************************************************

    for nbatch in range(num_batches):
        p_out = out[nbatch, :, :]

        if (valid_count[nbatch] < 1) or (nms_threshold <= 0) or (nms_threshold > 1):
            continue

        # sort and apply NMS
        nkeep = nms_topk if nms_topk<valid_count[nbatch] else int(valid_count[nbatch])
        # sort confidence in descend order and re-order output
        p_out[0:nkeep] = p_out[p_out[:, 1].topk(k=nkeep)]
        # p_out[nkeep:, 0] = -1    # not performed in original mxnet MultiBoxDetection, add by zhangxin

        # apply nms
        keep_indices = nms(p_out[0:nkeep], nms_threshold, force_suppress, num_classes)
        keep_indices = np.array(keep_indices)
        p_out[0:len(keep_indices)] = p_out[keep_indices]
        if len(keep_indices) < p_out.shape[0]:
            p_out[len(keep_indices):, 0] = -1
        out[nbatch, :, :] = p_out

    return out

def MaskRCNNKeypointBB8MultiBoxDetection(rois_concat, score_concat, cid_concat, maskrcnn_keypoint_cls_score_concat,
                    threshold=0.01, clip=True, background_id=0, nms_threshold=0.45, force_suppress=False,
                    nms_topk=400, name=None, im_info=(512,512,1)):
    """
    Parameters:
    :param rois_concat: batchsize x (num_post_nms x num_fpn) x 5
    :param score_concat: batchsize x (num_post_nms x num_fpn) x 1
    :param cid_concat: batchsize x (num_post_nms x num_fpn) x 1
    :param maskrcnn_keypoint_cls_score_concat: (batchsize x num_post_nms x num_fpn) x num_keypoints x granularity[0] x granularity[1]
    :param threshold: (float, optional, default=0.01) threshold to be a positive prediction
    :param clip: (boolean, optional, default=True) clip out-of-boundary boxes
    :param background_id: (int optional, default='0') background id
    :param nms_threshold: (float, optional, default=0.5) non-maximum suppression threshold
    :param force_suppress: (boolean, optional, default=False) suppress all detections regardless of class_id
    :param variances: (tuple of, optional, default=(0.1,0.1)) variances to be decoded from box regression output
    :param nms_topk: (int, optional, default=-1) keep maximum top k detections before nms, -1 for no limit.
    :param out: (NDArray, optional) the output NDArray to hold the result.
    :param name:

    :return: out: (NDArray or list of NDArray) the output of this function.
    """
    assert background_id == 0, "No implementation for background_id is not 0!!"
    # assert len(variances) == 2, "Variance size must be 2"
    assert nms_threshold > 0, "NMS_threshold should be greater than 0!!!"
    assert nms_threshold <=1, "NMS_threshold should be less than 1!!!"

    # ctx = cls_prob.context
    num_batches = score_concat.shape[0]
    num_rois_per_image = score_concat.shape[1]
    num_keypoints = maskrcnn_keypoint_cls_score_concat.shape[1]
    num_classes = int(np.max(cid_concat) + 1)
    granularity = tuple([maskrcnn_keypoint_cls_score_concat.shape[2], maskrcnn_keypoint_cls_score_concat.shape[3]])
    rois_concat[:, 1:5] /= np.array([im_info[1], im_info[0], im_info[1], im_info[0]])
    rois_concat = rois_concat.reshape((num_batches, num_rois_per_image, 5))
    maskrcnn_keypoint_cls_score_concat = np.reshape(maskrcnn_keypoint_cls_score_concat,
                                           newshape=(num_batches, num_rois_per_image, num_keypoints,
                                                    granularity[0],
                                                    granularity[1]))

    out = np.ones(shape=(num_batches, num_rois_per_image, 22)) * -1
    # remove background, restore original id
    out[:, :, 0:1] = cid_concat
    out[:, :, 1:2] = score_concat
    out[:, :, 2:6] = rois_concat[:, :, 1:5]
    out[:, :, 6:22] = TransformMaskRCNNKeypointBB8(rois_concat[:, :, 1:5], maskrcnn_keypoint_cls_score_concat, clip)
    out = mx.nd.array(out)

    # if the score < positive threshold, reset the id and score to -1
    out[:, :, 0] = mx.nd.where(condition=out[:, :, 1]<threshold,
                x=mx.nd.ones_like(out[:, :, 1]) * -1,
                y=out[:, :, 0])
    out[:, :, 1] = mx.nd.where(condition=out[:, :, 1] < threshold,
                               x=mx.nd.ones_like(out[:, :, 1]) * -1,
                               y=out[:, :, 1])

    valid_count = mx.nd.sum(out[:, :, 0] >= 0, axis=0, keepdims=False, exclude=True)
    valid_count = valid_count.asnumpy()

    #*******************************************************************************************

    for nbatch in range(num_batches):
        p_out = out[nbatch, :, :]

        if (valid_count[nbatch] < 1) or (nms_threshold <= 0) or (nms_threshold > 1):
            continue

        # sort and apply NMS
        nkeep = nms_topk if nms_topk<valid_count[nbatch] else int(valid_count[nbatch])
        # sort confidence in descend order and re-order output
        p_out[0:nkeep] = p_out[p_out[:, 1].topk(k=nkeep)]
        # p_out[nkeep:, 0] = -1    # not performed in original mxnet MultiBoxDetection, add by zhangxin

        # apply nms
        keep_indices = nms(p_out[0:nkeep], nms_threshold, force_suppress, num_classes)
        keep_indices = np.array(keep_indices)
        p_out[0:len(keep_indices)] = p_out[keep_indices]
        if len(keep_indices) < p_out.shape[0]:
            p_out[len(keep_indices):, 0] = -1
        out[nbatch, :, :] = p_out

    return out

def RCNNOffsetBB8MultiBoxDetection(rois_concat, score_concat, cid_concat,
                                       rcnn_bb8offset_pred_concat,
                    threshold=0.01, clip=True, background_id=0, nms_threshold=0.45, force_suppress=False,
                    bb8variances=(0.1, 0.1), nms_topk=400, name=None, im_info=(512,512,1)):
    """
    Parameters:
    :param rois_concat: batchsize x (num_post_nms) x 5
    :param score_concat: batchsize x (num_post_nms) x 1
    :param cid_concat: batchsize x (num_post_nms) x 1
    :param rcnn_bb8offset_pred_concat: (batchsize x num_post_nms) x (2 x num_keypoints)
    :param threshold: (float, optional, default=0.01) threshold to be a positive prediction
    :param clip: (boolean, optional, default=True) clip out-of-boundary boxes
    :param background_id: (int optional, default='0') background id
    :param nms_threshold: (float, optional, default=0.5) non-maximum suppression threshold
    :param force_suppress: (boolean, optional, default=False) suppress all detections regardless of class_id
    :param variances: (tuple of, optional, default=(0.1,0.1,0.2,0.2)) variances to be decoded from box regression output
    :param nms_topk: (int, optional, default=-1) keep maximum top k detections before nms, -1 for no limit.
    :param out: (NDArray, optional) the output NDArray to hold the result.
    :param name:

    :return: out: (NDArray or list of NDArray) the output of this function.
    """
    assert background_id == 0, "No implementation for background_id is not 0!!"
    assert len(bb8variances) == 2, "BB8Variance size must be 2"
    assert nms_threshold > 0, "NMS_threshold should be greater than 0!!!"
    assert nms_threshold <=1, "NMS_threshold should be less than 1!!!"

    # ctx = cls_prob.context
    num_batches = score_concat.shape[0]
    num_rois_per_image = score_concat.shape[1]
    num_keypoints = int(rcnn_bb8offset_pred_concat.shape[1] / 2)
    num_classes = int(np.max(cid_concat) + 1)

    rois_concat[:, 1:5] /= np.array([im_info[1], im_info[0], im_info[1], im_info[0]])
    rois_concat = rois_concat.reshape((num_batches, num_rois_per_image, 5))
    rcnn_bb8offset_pred_concat = np.reshape(rcnn_bb8offset_pred_concat,
                                           newshape=(num_batches, num_rois_per_image, num_keypoints, 2))

    out = np.ones(shape=(num_batches, num_rois_per_image, 22)) * -1
    # remove background, restore original id
    out[:, :, 0:1] = cid_concat
    out[:, :, 1:2] = score_concat
    out[:, :, 2:6] = rois_concat[:, :, 1:5]
    out[:, :, 6:22] = TransformRCNNOffsetBB8(rois_concat[:, :, 1:5], rcnn_bb8offset_pred_concat, clip, bb8variances)
    out = mx.nd.array(out)

    # if the score < positive threshold, reset the id and score to -1
    out[:, :, 0] = mx.nd.where(condition=out[:, :, 1]<threshold,
                x=mx.nd.ones_like(out[:, :, 1]) * -1,
                y=out[:, :, 0])
    out[:, :, 1] = mx.nd.where(condition=out[:, :, 1] < threshold,
                               x=mx.nd.ones_like(out[:, :, 1]) * -1,
                               y=out[:, :, 1])

    valid_count = mx.nd.sum(out[:, :, 0] >= 0, axis=0, keepdims=False, exclude=True)
    valid_count = valid_count.asnumpy()

    #*******************************************************************************************

    for nbatch in range(num_batches):
        p_out = out[nbatch, :, :]

        if (valid_count[nbatch] < 1) or (nms_threshold <= 0) or (nms_threshold > 1):
            continue

        # sort and apply NMS
        nkeep = nms_topk if nms_topk<valid_count[nbatch] else int(valid_count[nbatch])
        # sort confidence in descend order and re-order output
        p_out[0:nkeep] = p_out[p_out[:, 1].topk(k=nkeep)]
        # p_out[nkeep:, 0] = -1    # not performed in original mxnet MultiBoxDetection, add by zhangxin

        # apply nms
        keep_indices = nms(p_out[0:nkeep], nms_threshold, force_suppress, num_classes)
        keep_indices = np.array(keep_indices)
        p_out[0:len(keep_indices)] = p_out[keep_indices]
        if len(keep_indices) < p_out.shape[0]:
            p_out[len(keep_indices):, 0] = -1
        out[nbatch, :, :] = p_out

    return out













