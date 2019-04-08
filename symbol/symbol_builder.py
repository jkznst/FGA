import mxnet as mx
from symbol.common import multi_layer_feature_SSD, multibox_layer_FPN, multibox_layer_SSD, multibox_layer_FPN_RCNN, multibox_layer_SSD_RCNN
# from operator_py.training_target import *
from operator_py.rpn_proposal import *
# from operator_py.proposal_FGAtarget_cls_softmax_reg_offset import *
# from operator_py.proposal_target_bb8offset_reg import *
# from operator_py.proposal_target_maskrcnn_keypoint import *
# from operator_py.proposal_target_heatmap import *
# from operator_py.proposal_target_boundary_offset import *
from operator_py.proposal_target_boundary_offset_soft_cls import *
# from operator_py.weightedCELoss import *
from operator_py.softCELoss import *

def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)


def training_targets(anchors, class_preds, labels):
    # labels_np = labels.asnumpy()
    # view_cls_label = mx.nd.slice_axis(data=labels, axis=2, begin=6, end=7)
    # inplane_cls_label = mx.nd.slice_axis(data=labels, axis=2, begin=7, end=8)
    # bbox_label = mx.nd.slice_axis(data=labels, axis=2, begin=1, end=5)
    # label_valid_count = mx.symbol.sum(mx.symbol.slice_axis(labels, axis=2, begin=0, end=1) >= 0, axis=1)
    # class_preds = class_preds.transpose(axes=(0,2,1))

    box_target, box_mask, cls_target = mx.symbol.contrib.MultiBoxTarget(anchors, labels, class_preds, overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")

    anchor_mask = box_mask.reshape(shape=(0, -1, 4))    # batchsize x num_anchors x 4
    bb8_mask = mx.symbol.repeat(data=anchor_mask, repeats=4, axis=2)  # batchsize x num_anchors x 16
    #anchor_mask = mx.nd.mean(data=anchor_mask, axis=2, keepdims=False, exclude=False)

    anchors_in_use = mx.symbol.broadcast_mul(lhs=anchor_mask,rhs=anchors)   # batchsize x num_anchors x 4

    # transform the anchors from [xmin, ymin, xmax, ymax] to [cx, cy, wx, hy]

    centerx = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1) + \
               mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3)) / 2
    centery = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2) + \
               mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4)) / 2
    width = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3) - \
               mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1)) + 0.0000001
    height = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4) - \
               mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2)) + 0.0000001
    # anchors_in_use_transformed = mx.symbol.zeros_like(data=anchors_in_use)
    # anchors_in_use_transformed[:, :, 0] = (anchors_in_use[:, :, 0] + anchors_in_use[:, :, 2]) / 2
    # anchors_in_use_transformed[:, :, 1] = (anchors_in_use[:, :, 1] + anchors_in_use[:, :, 3]) / 2
    # anchors_in_use_transformed[:, :, 2] = anchors_in_use[:, :, 2] - anchors_in_use[:, :, 0] + 0.0000001
    # anchors_in_use_transformed[:, :, 3] = anchors_in_use[:, :, 3] - anchors_in_use[:, :, 1] + 0.0000001
    anchors_in_use_transformed = mx.symbol.concat(centerx, centery, width, height, dim=2)   # batchsize x num_anchors x 4

    bb8_target = mx.symbol.zeros_like(bb8_mask)
    bb8_label = mx.symbol.slice_axis(data=labels, axis=2, begin=8, end=24)
    # cls_target_temp = mx.symbol.repeat(data=cls_target, repeats=4, axis=1)
    # cls_target_temp = mx.symbol.reshape(data=cls_target_temp, shape=(0, -1, 4)) # batchsize x num_anchors x 4
    # calculate targets for OCCLUSION dataset
    for cid in range(1,9):
        # cid_target_mask = (cls_target == cid)
        # cid_target_mask = mx.symbol.reshape(data=cid_target_mask, shape=(0,-1,1))   # batchsize x num_anchors x 1
        # cid_anchors_in_use_transformed = mx.symbol.broadcast_mul(lhs=cid_target_mask, rhs=anchors_in_use_transformed)   # batchsize x num_anchors x 4
        cid_anchors_in_use_transformed = mx.symbol.where(condition=(cls_target==cid), x=anchors_in_use_transformed,
                                                         y=mx.symbol.zeros_like(anchors_in_use_transformed))
        cid_label_mask = (mx.symbol.slice_axis(data=labels, axis=2, begin=0, end=1) == cid-1)
        cid_bb8_label = mx.symbol.broadcast_mul(lhs=cid_label_mask, rhs=bb8_label)
        cid_bb8_label = mx.symbol.max(cid_bb8_label, axis=1, keepdims=True) # batchsize x 1 x 16

        # substract center
        cid_bb8_target = mx.symbol.broadcast_sub(cid_bb8_label, mx.symbol.tile(   # repeat single element !! error
            data=mx.symbol.slice_axis(cid_anchors_in_use_transformed, axis=2, begin=0, end=2),
            reps=(1,1,8)))
        # divide by w and h
        cid_bb8_target = mx.symbol.broadcast_div(cid_bb8_target, mx.symbol.tile(
            data=mx.symbol.slice_axis(cid_anchors_in_use_transformed, axis=2, begin=2, end=4),
            reps=(1, 1, 8))) / 0.1  # variance
        # cid_bb8_target = mx.symbol.broadcast_mul(lhs=cid_target_mask, rhs=cid_bb8_target)   # this sentence will cause loss explosion, don't know why
        # cid_bb8_target = mx.symbol.where(condition=(mx.symbol.repeat(cls_target_temp, repeats=4, axis=2)==cid), x=cid_bb8_target,
        #                                  y=mx.symbol.zeros_like(cid_bb8_target))
        cid_bb8_target = mx.symbol.where(condition=(cls_target == cid),
                                         x=cid_bb8_target,
                                         y=mx.symbol.zeros_like(cid_bb8_target))
        bb8_target = bb8_target + cid_bb8_target

    condition = bb8_mask > 0.5
    bb8_target = mx.symbol.where(condition=condition, x=bb8_target, y=mx.symbol.zeros_like(data=bb8_target))

    bb8_target = bb8_target.flatten()   # batchsize x (num_anchors x 16)
    bb8_mask = bb8_mask.flatten()       # batchsize x (num_anchors x 16)
    return box_target, box_mask, cls_target, bb8_target, bb8_mask


def get_symbol_train(network, num_classes, alpha_bb8, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, minimum_negative_samples=0, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    label = mx.sym.Variable('label')
    body = import_module(network).get_symbol(num_classes=num_classes, **kwargs)

    layers = multi_layer_feature_SSD(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_SSD(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_resnet_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnet import get_ssd_conv, get_ssd_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_FPN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_resnetd_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnet import get_detnet_conv, get_detnet_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_detnet_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_detnet_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_FPN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_resnetdeeplabv2_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnet import get_deeplabv2_conv, get_detnet_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_deeplabv2_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_detnet_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_FPN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_resnetm_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_FPN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_RCNN_resnetm_fpn_train_old(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    from rcnn.config import config
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    conv_fpn_feat_dict, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    # rpn_bbox_pred : (N, 4 x num_anchors, H, W)
    # rpn_cls_pred : (N, num_anchor x (num_classes + 1), H, W)
    # rpn_anchor: (N, num_all_anchor, 4)
    rpn_bbox_pred_dict, rpn_cls_prob_dict, rpn_anchor_dict, \
    loc_preds, cls_preds, anchor_boxes = multibox_layer_FPN_RCNN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
    #                                                                                         name="training_targets",
    #                                                                                         anchors=anchor_boxes,
    #                                                                                         cls_preds=cls_preds,
    #                                                                                         labels=label)

    rpn_targets = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = rpn_targets[0]
    loc_target_mask = rpn_targets[1]
    cls_target = rpn_targets[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    # bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
    #     data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    # bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
    #     normalization='valid', name="bb8_loss")

    # rpn proposal
    im_info = (512, 512, 1)    # (H, W, scale)
    feat_stride = [4, 8, 16, 32, 64]
    granularity = (3, 3)
    FGA_cls_target_list = []
    FGA_reg_target_list = []
    FGA_reg_weight_list = []
    rcnn_cls_score_list = []
    rcnn_bb8_pred_list = []

    # # shared parameters for predictions
    rcnn_head_conv_weight = mx.symbol.Variable('rcnn_head_conv_weight')
    rcnn_head_conv_bias = mx.symbol.Variable('rcnn_head_conv_bias')
    rcnn_FGA_cls_weight = mx.symbol.Variable('rcnn_FGA_cls_weight')
    rcnn_FGA_cls_bias = mx.symbol.Variable('rcnn_FGA_cls_bias')
    rcnn_FGA_reg_weight = mx.symbol.Variable('rcnn_FGA_reg_weight')
    rcnn_FGA_reg_bias = mx.symbol.Variable('rcnn_FGA_reg_bias')

    for s in feat_stride:
        rpn_det = mx.contrib.symbol.MultiBoxDetection(*[rpn_cls_prob_dict['cls_prob_stride%s'% s],
                                                    rpn_bbox_pred_dict['bbox_pred_stride%s' % s],
                                                    rpn_anchor_dict['rpn_anchor_stride%s' % s]],
                                              name="rpn_proposal_stride%s" % s, nms_threshold=0.45, force_suppress=False,
                                              variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
        # rpn_det = mx.symbol.MakeLoss(data=rpn_det, grad_scale=0, name="rpn_det_out_stride64")
        rois, score, cid = mx.symbol.Custom(op_type='rpn_proposal',
                         rpn_det=rpn_det,
                         output_score=True,
                         rpn_post_nms_top_n=400, im_info=im_info
                         )
        rois = mx.symbol.reshape(rois, shape=(-1, 5))
        # rois = mx.symbol.MakeLoss(data=rois, grad_scale=0, name='rpn_roi')
        # score = mx.symbol.MakeLoss(data=score, grad_scale=0, name='rpn_score')
        # cid = mx.symbol.MakeLoss(data=cid, grad_scale=0, name='rpn_cid')
        #     rpn_proposal_dict.update({'rpn_proposal_stride%s' % s: rpn_det})

        # rcnn roi proposal target
        group = mx.symbol.Custom(rois=rois, gt_boxes=label, op_type='bb8_proposal_target',
                             num_keypoints=8, batch_images=2,
                             batch_rois=128, fg_fraction=1.0,
                             fg_overlap=0.5, bb8_variance=(0.1, 0.1),
                             im_info=im_info, granularity=granularity)
        rois = group[0]
        FGA_cls_target = group[1]
        FGA_reg_target = group[2]
        FGA_reg_weight = group[3]

        FGA_cls_target_list.append(FGA_cls_target)
        FGA_reg_target_list.append(FGA_reg_target)
        FGA_reg_weight_list.append(FGA_reg_weight)

        # rcnn roi pool
        roi_pool = mx.symbol.ROIPooling(
            name='roi_pool_stride{}'.format(s), data=conv_fpn_feat_dict['stride%s' % s], rois=rois, pooled_size=granularity,
            spatial_scale=1.0 / s)
        # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0, name='roi_pool')

        head_conv = mx.symbol.Convolution(data=roi_pool, weight=rcnn_head_conv_weight, bias=rcnn_head_conv_bias,
                                          kernel=(3, 3), stride=(1,1), pad=(1,1), num_filter=256, name="rcnn_head_conv_stride{}".format(s))
        head_relu = mx.symbol.Activation(head_conv, act_type='relu', name="rcnn_head_relu_stride{}".format(s))

        rcnn_cls_score = mx.symbol.Convolution(data=head_relu, weight=rcnn_FGA_cls_weight, bias=rcnn_FGA_cls_bias,
                                               kernel=(3, 3), stride=(1,1), pad=(1,1), num_filter=8, name="rcnn_FGA_cls_score_stride{}".format(s))
        rcnn_bb8_pred = mx.symbol.Convolution(data=head_relu, weight=rcnn_FGA_reg_weight, bias=rcnn_FGA_reg_bias,
                                              kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                               num_filter=16, name="rcnn_FGA_bb8_pred_stride{}".format(s))
        rcnn_cls_score_list.append(rcnn_cls_score)
        rcnn_bb8_pred_list.append(rcnn_bb8_pred)

    # concat output of each level
    rcnn_FGA_cls_score_concat = mx.symbol.concat(*rcnn_cls_score_list, dim=0)  # [num_rois_4level, num_keypoints, granularity[0], granularity[1]]
    rcnn_FGA_bb8_pred_concat = mx.symbol.concat(*rcnn_bb8_pred_list, dim=0)  # [num_rois_4level, num_keypoints*2, granularity[0], granularity[1]]

    FGA_cls_target_concat = mx.symbol.concat(*FGA_cls_target_list, dim=0)
    FGA_reg_target_concat = mx.symbol.concat(*FGA_reg_target_list, dim=0)
    FGA_reg_weight_concat = mx.symbol.concat(*FGA_reg_weight_list, dim=0)

    # loss
    rcnn_FGA_cls_prob = mx.symbol.LogisticRegressionOutput(data=rcnn_FGA_cls_score_concat, label=FGA_cls_target_concat,
                                       grad_scale=1.0 / 640., name='rcnn_FGA_cls_prob')

    rcnn_FGA_bb8_reg_loss_ = FGA_reg_weight_concat * mx.symbol.smooth_l1(name='rcnn_FGA_bb8_reg_loss_', scalar=1.0,
                                                   data=(rcnn_FGA_bb8_pred_concat - FGA_reg_target_concat))

    rcnn_FGA_bb8_reg_loss = mx.sym.MakeLoss(name='rcnn_FGA_bb8_reg_loss', data=rcnn_FGA_bb8_reg_loss_, grad_scale=1.0 / 640.)
    rcnn_group = [rcnn_FGA_cls_prob, rcnn_FGA_bb8_reg_loss]
    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    rcnn_FGA_cls_target = mx.symbol.MakeLoss(data=FGA_cls_target_concat, grad_scale=0, name="rcnn_FGA_cls_target")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    # anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    # loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
    #                              grad_scale=0, name='loc_mae')
    # loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    # loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    # bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    # bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    # bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    # bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
    #                              grad_scale=0, name='bb8_mae')

    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=0.45, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    # loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    # out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
    #                        anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    # out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det, *rcnn_group, rcnn_FGA_cls_target])
    return out

def get_RCNN_offset_resnetm_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], im_info=(), **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    from rcnn.config import config
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    conv_fpn_feat_dict, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    # rpn_bbox_pred : (N, 4 x num_anchors, H, W)
    # rpn_cls_pred : (N, num_anchor x (num_classes + 1), H, W)
    # rpn_anchor: (N, num_all_anchor, 4)
    rpn_bbox_pred_dict, rpn_cls_prob_dict, rpn_anchor_dict, \
    loc_preds, cls_preds, anchor_boxes = multibox_layer_FPN_RCNN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    rpn_targets = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = rpn_targets[0]
    loc_target_mask = rpn_targets[1]
    cls_target = rpn_targets[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1.0, multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1.0, \
        normalization='valid', name="loc_loss")

    # rpn proposal
    # im_info = (512, 512, 1)    # (H, W, scale)
    # rpn_feat_stride = [4, 8, 16, 32, 64]
    # granularity = (56, 56)

    # rpn detection results merging all the levels, set a higher nms threshold to keep more proposals
    rpn_det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="rpn_proposal", nms_threshold=0.7, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)

    # # select foreground region proposals, and transform the coordinate from [0,1] to [0, 448]
    rois, score, cid = mx.symbol.Custom(op_type='rpn_proposal',
                     rpn_det=rpn_det,
                     output_score=True,
                     rpn_post_nms_top_n=400, im_info=im_info
                     )
    rois = mx.symbol.reshape(rois, shape=(-1, 5))
    # rois = mx.symbol.MakeLoss(data=rois, grad_scale=0, name='rpn_roi')
    # score = mx.symbol.MakeLoss(data=score, grad_scale=0, name='rpn_score')
    # cid = mx.symbol.MakeLoss(data=cid, grad_scale=0, name='rpn_cid')
    #     rpn_proposal_dict.update({'rpn_proposal_stride%s' % s: rpn_det})

    # rcnn roi proposal target
    group = mx.symbol.Custom(rois=rois, gt_boxes=label, op_type='bb8_proposal_target_offset_reg',
                         num_keypoints=8, batch_images=2,
                         batch_rois=256, fg_fraction=1.0,
                         fg_overlap=0.8, bb8_variance=(0.1, 0.1),
                         im_info=im_info)
    rois = group[0]
    rcnn_bb8offset_reg_target = group[1]
    rcnn_bb8offset_reg_weight = group[2]

    # # rcnn roi pool
    roi_pool = mx.symbol.contrib.ROIAlign(
        name='roi_pool', data=conv_fpn_feat_dict['stride8'], rois=rois, pooled_size=(7, 7),
        spatial_scale=1.0 / 8.)
    # roi_pool = mx.symbol.Custom(op_type="fpn_roi_pool",
    #                             rcnn_strides="(16,8,4)",
    #                             pool_h=7, pool_w=7,
    #                             feat_stride16=conv_fpn_feat_dict['stride16'],
    #                             feat_stride8=conv_fpn_feat_dict['stride8'],
    #                             feat_stride4=conv_fpn_feat_dict['stride4'],
    #                             rois=rois)
    # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0., name='roi_pool')

    # # bb8 offset regression head
    flatten = mx.symbol.flatten(data=roi_pool, name="rcnn_bb8offset_reg_flatten")
    fc6_weight = mx.symbol.Variable(name="rcnn_bb8offset_reg_fc6_weight", init=mx.init.Xavier())
    fc6_bias = mx.symbol.Variable(name="rcnn_bb8offset_reg_fc6_bias", init=mx.init.Constant(0.0))
    fc6 = mx.symbol.FullyConnected(data=flatten, weight=fc6_weight, bias=fc6_bias, num_hidden=1024, name="rcnn_bb8offset_reg_fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_bb8offset_reg_relu6")
    fc7_weight = mx.symbol.Variable(name="rcnn_bb8offset_reg_fc7_weight", init=mx.init.Xavier())
    fc7_bias = mx.symbol.Variable(name="rcnn_bb8offset_reg_fc7_bias", init=mx.init.Constant(0.0))
    fc7 = mx.symbol.FullyConnected(data=relu6, weight=fc7_weight, bias=fc7_bias, num_hidden=1024, name="rcnn_bb8offset_reg_fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_bb8offset_reg_relu7")

    pred_weight = mx.symbol.Variable(name="rcnn_bb8offset_reg_pred_weight", init=mx.init.Normal(sigma=0.001))
    pred_bias = mx.symbol.Variable(name="rcnn_bb8offset_reg_pred_bias", init=mx.init.Constant(0.0))
    rcnn_bb8offset_reg_pred = mx.symbol.FullyConnected(data=relu7, weight=pred_weight, bias=pred_bias,
                                                   num_hidden=16, name="rcnn_bb8offset_reg_pred")

    # keypoint offset loss
    rcnn_bb8offset_reg_loss_ = mx.symbol.smooth_l1(name="rcnn_bb8offset_reg_loss_", \
                                    data=rcnn_bb8offset_reg_weight * (rcnn_bb8offset_reg_pred - rcnn_bb8offset_reg_target), scalar=1.0)
    rcnn_bb8offset_reg_loss = mx.symbol.MakeLoss(rcnn_bb8offset_reg_loss_, grad_scale=alpha_bb8, \
                                  normalization='valid', name="rcnn_bb8offset_reg_loss")


    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    rpn_loc_target = mx.symbol.MakeLoss(data=loc_target * loc_target_mask, grad_scale=0, name="rpn_loc_target")
    det_out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
                                                  name="rpn_det_out", nms_threshold=0.45, force_suppress=False,
                                                  variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
    det = mx.symbol.MakeLoss(data=det_out, grad_scale=0, name="det_out")
    # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0., name="roi_pool_monitor")

    rois = mx.symbol.MakeLoss(data=rois, grad_scale=0, name="rois")
    score = mx.symbol.MakeLoss(data=score, grad_scale=0, name="score")
    cid = mx.symbol.MakeLoss(data=cid, grad_scale=0, name="cid")
    #
    rcnn_bb8offset_reg_target_monitor = mx.symbol.MakeLoss(data=rcnn_bb8offset_reg_weight * rcnn_bb8offset_reg_target, grad_scale=0,
                                             name="rcnn_bb8offset_reg_target_monitor")

    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, rpn_loc_target, det,
                           rois, score, cid,
                           rcnn_bb8offset_reg_target_monitor, rcnn_bb8offset_reg_loss
                           ])
    return out

def get_RCNN_offset_resnetm_fpn_test(num_classes, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], im_info=(), **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    from rcnn.config import config
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    conv_fpn_feat_dict, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    # rpn_bbox_pred : (N, 4 x num_anchors, H, W)
    # rpn_cls_pred : (N, num_anchor x (num_classes + 1), H, W)
    # rpn_anchor: (N, num_all_anchor, 4)
    rpn_bbox_pred_dict, rpn_cls_prob_dict, rpn_anchor_dict, \
    loc_preds, cls_preds, anchor_boxes = multibox_layer_FPN_RCNN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    cls_prob = mx.symbol.softmax(data=cls_preds, axis=1, name="rpn_cls_prob")

    # rpn proposal
    # im_info = (512, 512, 1)    # (H, W, scale)
    # rpn_feat_stride = [4, 8, 16, 32, 64]

    # rpn detection results merging all the levels, set a higher nms threshold to keep more proposals
    rpn_det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="rpn_proposal", nms_threshold=0.45, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)

    # # select foreground region proposals, and transform the coordinate from [0,1] to [0, 448]
    rois, score, cid = mx.symbol.Custom(op_type='rpn_proposal',
                     rpn_det=rpn_det,
                     output_score=True,
                     rpn_post_nms_top_n=400, im_info=im_info
                     )
    rois = mx.symbol.reshape(rois, shape=(-1, 5))

    # # rcnn roi pool
    roi_pool = mx.symbol.contrib.ROIAlign(
        name='roi_pool', data=conv_fpn_feat_dict['stride8'], rois=rois, pooled_size=(7, 7),
        spatial_scale=1.0 / 8.)
    # roi_pool = mx.symbol.Custom(op_type="fpn_roi_pool",
    #                             rcnn_strides="(16,8,4)",
    #                             pool_h=7, pool_w=7,
    #                             feat_stride16=conv_fpn_feat_dict['stride16'],
    #                             feat_stride8=conv_fpn_feat_dict['stride8'],
    #                             feat_stride4=conv_fpn_feat_dict['stride4'],
    #                             rois=rois)
    # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0., name='roi_pool')

    # # bb8 offset regression head
    flatten = mx.symbol.flatten(data=roi_pool, name="rcnn_bb8offset_reg_flatten")
    fc6_weight = mx.symbol.Variable(name="rcnn_bb8offset_reg_fc6_weight", init=mx.init.Xavier())
    fc6_bias = mx.symbol.Variable(name="rcnn_bb8offset_reg_fc6_bias", init=mx.init.Constant(0.0))
    fc6 = mx.symbol.FullyConnected(data=flatten, weight=fc6_weight, bias=fc6_bias, num_hidden=1024, name="rcnn_bb8offset_reg_fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_bb8offset_reg_relu6")
    fc7_weight = mx.symbol.Variable(name="rcnn_bb8offset_reg_fc7_weight", init=mx.init.Xavier())
    fc7_bias = mx.symbol.Variable(name="rcnn_bb8offset_reg_fc7_bias", init=mx.init.Constant(0.0))
    fc7 = mx.symbol.FullyConnected(data=relu6, weight=fc7_weight, bias=fc7_bias, num_hidden=1024, name="rcnn_bb8offset_reg_fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_bb8offset_reg_relu7")

    pred_weight = mx.symbol.Variable(name="rcnn_bb8offset_reg_pred_weight", init=mx.init.Normal(sigma=0.001))
    pred_bias = mx.symbol.Variable(name="rcnn_bb8offset_reg_pred_bias", init=mx.init.Constant(0.0))
    rcnn_bb8offset_reg_pred = mx.symbol.FullyConnected(data=relu7, weight=pred_weight, bias=pred_bias,
                                                   num_hidden=16, name="rcnn_bb8offset_reg_pred")


    # monitoring training status
    # det = mx.symbol.MakeLoss(data=rpn_det, grad_scale=0, name="det_out")
    # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0., name="roi_pool_monitor")

    rois = mx.symbol.MakeLoss(data=rois, grad_scale=0, name="rois")
    score = mx.symbol.MakeLoss(data=score, grad_scale=0, name="score")
    cid = mx.symbol.MakeLoss(data=cid, grad_scale=0, name="cid")
    #
    rcnn_bb8offset_reg_pred_out = mx.symbol.MakeLoss(data=rcnn_bb8offset_reg_pred, grad_scale=0,
                                             name="rcnn_bb8offset_reg_pred_out")

    out = mx.symbol.Group([rois, score, cid,
                           rcnn_bb8offset_reg_pred_out])
    return out

def get_MaskRCNN_keypoint_resnetm_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], im_info=(512, 512, 1), **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    from rcnn.config import config
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    conv_fpn_feat_dict, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    # rpn_bbox_pred : (N, 4 x num_anchors, H, W)
    # rpn_cls_pred : (N, num_anchor x (num_classes + 1), H, W)
    # rpn_anchor: (N, num_all_anchor, 4)
    rpn_bbox_pred_dict, rpn_cls_prob_dict, rpn_anchor_dict, \
    loc_preds, cls_preds, anchor_boxes = multibox_layer_FPN_RCNN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    rpn_targets = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = rpn_targets[0]
    loc_target_mask = rpn_targets[1]
    cls_target = rpn_targets[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1.0, multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1.0, \
        normalization='valid', name="loc_loss")

    # rpn proposal
    # im_info = (800, 800, 1)    # (H, W, scale)
    rpn_feat_stride = [4, 8, 16, 32, 64]
    granularity = (56, 56)
    num_keypoints = 8

    # rpn detection results merging all the levels, set a higher nms threshold to keep more proposals
    rpn_det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="rpn_proposal", nms_threshold=0.7, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)

    # # select foreground region proposals, and transform the coordinate from [0,1] to [0, 448]
    rois, score, cid = mx.symbol.Custom(op_type='rpn_proposal',
                     rpn_det=rpn_det,
                     output_score=True,
                     rpn_post_nms_top_n=400, im_info=im_info
                     )
    rois = mx.symbol.reshape(rois, shape=(-1, 5))

    # rcnn roi proposal target
    group = mx.symbol.Custom(rois=rois, gt_boxes=label, op_type='proposal_target_maskrcnn_keypoint',
                         num_keypoints=num_keypoints, batch_images=2,
                         batch_rois=256, fg_fraction=1.0,
                         fg_overlap=0.8, bb8_variance=(0.1, 0.1),
                         im_info=im_info, mask_shape=granularity)
    rois = group[0]
    maskrcnn_keypoint_cls_target = group[1]

    # # rcnn roi pool
    roi_pool = mx.symbol.contrib.ROIAlign(
        name='roi_pool', data=conv_fpn_feat_dict['stride8'], rois=rois, pooled_size=(14, 14),
        spatial_scale=1.0 / 8.)

    # # Mask RCNN head
    for i in range(8):
        weight = mx.symbol.Variable(name="maskrcnn_keypoint_cls_conv{}_weight".format(i+1),
                              init=mx.init.MSRAPrelu(factor_type="out", slope=0.0))
        bias = mx.symbol.Variable(name="maskrcnn_keypoint_cls_conv{}_bias".format(i+1),
                              init=mx.init.Constant(0.0))
        x = mx.symbol.Convolution(data=roi_pool, weight=weight, bias=bias,
                                      kernel=(3, 3), stride=(1,1), pad=(1,1), num_filter=512, name="maskrcnn_keypoint_cls_conv{}".format(i + 1))
        x = mx.symbol.Activation(data=x, act_type="relu", name="maskrcnn_keypoint_cls_relu{}".format(i + 1))

    deconv_weight = mx.symbol.Variable(name="maskrcnn_keypoint_cls_deconv_weight", init=mx.init.MSRAPrelu(factor_type="out", slope=0.0))
    deconv_bias = mx.symbol.Variable(name="maskrcnn_keypoint_cls_deconv_bias", init=mx.init.Constant(0.0))
    x = mx.symbol.Deconvolution(data=x, weight=deconv_weight, bias=deconv_bias,
                                kernel=(4,4), stride=(2,2), pad=(1,1), target_shape=(28, 28),
                                num_filter=num_keypoints, no_bias=False, name="maskrcnn_keypoint_cls_deconv")

    maskrcnn_keypoint_cls_score = mx.symbol.contrib.BilinearResize2D(data=x, height=56, width=56, name="maskrcnn_keypoint_cls_score")

    # # maskrcnn keypoint loss
    maskrcnn_keypoint_cls_prob = mx.symbol.SoftmaxOutput(data=maskrcnn_keypoint_cls_score.reshape((0,num_keypoints,-1)),
                                                         label=maskrcnn_keypoint_cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1.0, preserve_shape=True, \
        normalization='valid', name="maskrcnn_keypoint_cls_prob")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # rpn_loc_target = mx.symbol.MakeLoss(data=loc_target * loc_target_mask, grad_scale=0, name="rpn_loc_target")
    det_out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
                                                  name="rpn_det_out", nms_threshold=0.45, force_suppress=False,
                                                  variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
    det = mx.symbol.MakeLoss(data=det_out, grad_scale=0, name="det_out")
    # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0., name="roi_pool_monitor")

    # rois = mx.symbol.MakeLoss(data=rois, grad_scale=0, name="rois")
    # score = mx.symbol.MakeLoss(data=score, grad_scale=0, name="score")
    # cid = mx.symbol.MakeLoss(data=cid, grad_scale=0, name="cid")
    #
    maskrcnn_keypoint_cls_target_monitor = mx.symbol.MakeLoss(data=maskrcnn_keypoint_cls_target, grad_scale=0, name="maskrcnn_keypoint_cls_target")
    # rcnn_FGA_reg_target = mx.symbol.MakeLoss(data=FGA_reg_target, grad_scale=0, name="rcnn_FGA_reg_target")

    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det,
                           maskrcnn_keypoint_cls_target_monitor, maskrcnn_keypoint_cls_prob
                           ])
    return out

def get_MaskRCNN_keypoint_resnetm_fpn_test(num_classes, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], im_info=(512,512,1), **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    im_info: tuple (image_shape, image_shape, scale)
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    from rcnn.config import config
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    conv_fpn_feat_dict, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    # rpn_bbox_pred : (N, 4 x num_anchors, H, W)
    # rpn_cls_pred : (N, num_anchor x (num_classes + 1), H, W)
    # rpn_anchor: (N, num_all_anchor, 4)
    rpn_bbox_pred_dict, rpn_cls_prob_dict, rpn_anchor_dict, \
    loc_preds, cls_preds, anchor_boxes = multibox_layer_FPN_RCNN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.softmax(data=cls_preds, axis=1, name="rpn_cls_prob")

    # rpn proposal
    # im_info = (512, 512, 1)    # (H, W, scale)
    feat_stride = [4, 8, 16, 32, 64]
    granularity = (56, 56)
    num_keypoints = 8

    rpn_det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="rpn_proposal", nms_threshold=0.45, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)

    rois, score, cid = mx.symbol.Custom(op_type='rpn_proposal',
                     rpn_det=rpn_det,
                     output_score=True,
                     rpn_post_nms_top_n=400, im_info=im_info
                     )

    rois = mx.symbol.reshape(rois, shape=(-1, 5))

    # rcnn roi pool
    roi_pool = mx.symbol.contrib.ROIAlign(
        name='roi_pool', data=conv_fpn_feat_dict['stride8'], rois=rois, pooled_size=(14, 14),
        spatial_scale=1.0 / 8.)

    # # Mask RCNN head
    for i in range(8):
        weight = mx.symbol.Variable(name="maskrcnn_keypoint_cls_conv{}_weight".format(i+1),
                              init=mx.init.MSRAPrelu(factor_type="out", slope=0.0))
        bias = mx.symbol.Variable(name="maskrcnn_keypoint_cls_conv{}_bias".format(i+1),
                              init=mx.init.Constant(0.0))
        x = mx.symbol.Convolution(data=roi_pool, weight=weight, bias=bias,
                                      kernel=(3, 3), stride=(1,1), pad=(1,1), num_filter=512, name="maskrcnn_keypoint_cls_conv{}".format(i + 1))
        x = mx.symbol.Activation(data=x, act_type="relu", name="maskrcnn_keypoint_cls_relu{}".format(i + 1))

    deconv_weight = mx.symbol.Variable(name="maskrcnn_keypoint_cls_deconv_weight", init=mx.init.MSRAPrelu(factor_type="out", slope=0.0))
    deconv_bias = mx.symbol.Variable(name="maskrcnn_keypoint_cls_deconv_bias", init=mx.init.Constant(0.0))
    x = mx.symbol.Deconvolution(data=x, weight=deconv_weight, bias=deconv_bias,
                                kernel=(4,4), stride=(2,2), pad=(1,1), target_shape=(28, 28),
                                num_filter=num_keypoints, no_bias=False, name="maskrcnn_keypoint_cls_deconv")

    maskrcnn_keypoint_cls_score = mx.symbol.contrib.BilinearResize2D(data=x, height=56, width=56, name="maskrcnn_keypoint_cls_score")

    out = mx.symbol.Group([rois, score, cid, maskrcnn_keypoint_cls_score])
    return out

def get_FGARCNN_cls_softmax_reg_offset_resnetm_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], im_info=(), **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    conv_fpn_feat_dict, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    # rpn_bbox_pred : (N, 4 x num_anchors, H, W)
    # rpn_cls_pred : (N, num_anchor x (num_classes + 1), H, W)
    # rpn_anchor: (N, num_all_anchor, 4)
    rpn_bbox_pred_dict, rpn_cls_prob_dict, rpn_anchor_dict, \
    loc_preds, cls_preds, anchor_boxes = multibox_layer_FPN_RCNN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    rpn_targets = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = rpn_targets[0]
    loc_target_mask = rpn_targets[1]
    cls_target = rpn_targets[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1.0, multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1.0, \
        normalization='valid', name="loc_loss")

    # rpn proposal
    # im_info = (512, 512, 1)    # (H, W, scale)
    rpn_feat_stride = [4, 8, 16, 32, 64]
    granularity = (7, 7)

    # rpn detection results merging all the levels, set a higher nms threshold to keep more proposals
    rpn_det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="rpn_proposal", nms_threshold=0.7, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)

    # # select foreground region proposals, and transform the coordinate from [0,1] to [0, 448]
    rois, score, cid = mx.symbol.Custom(op_type='rpn_proposal',
                     rpn_det=rpn_det,
                     output_score=True,
                     rpn_post_nms_top_n=400, im_info=im_info
                     )
    rois = mx.symbol.reshape(rois, shape=(-1, 5))
    # rois = mx.symbol.MakeLoss(data=rois, grad_scale=0, name='rpn_roi')
    # score = mx.symbol.MakeLoss(data=score, grad_scale=0, name='rpn_score')
    # cid = mx.symbol.MakeLoss(data=cid, grad_scale=0, name='rpn_cid')
    #     rpn_proposal_dict.update({'rpn_proposal_stride%s' % s: rpn_det})

    # rcnn roi proposal target
    group = mx.symbol.Custom(rois=rois, gt_boxes=label, op_type='bb8_proposal_FGAtarget_cls_softmax_reg_offset',
                         num_keypoints=8, batch_images=2,
                         batch_rois=256, fg_fraction=1.0,
                         fg_overlap=0.8, bb8_variance=(0.2, 0.2),
                         im_info=im_info, granularity=granularity)
    rois = group[0]
    FGA_cls_target = group[1]    # (N, 8)
    FGA_reg_target = group[2]    # (N, 16)
    FGA_reg_weight = group[3]

    # # rcnn roi pool
    roi_pool = mx.symbol.contrib.ROIAlign(
        name='roi_pool', data=conv_fpn_feat_dict['stride8'], rois=rois, pooled_size=(7, 7),
        spatial_scale=1.0 / 8.)
    # roi_pool = mx.symbol.Custom(op_type="fpn_roi_pool",
    #                             rcnn_strides="(16,8,4)",
    #                             pool_h=14, pool_w=14,
    #                             feat_stride16=conv_fpn_feat_dict['stride16'],
    #                             feat_stride8=conv_fpn_feat_dict['stride8'],
    #                             feat_stride4=conv_fpn_feat_dict['stride4'],
    #                             rois=rois)
    # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0., name='roi_pool')

    # bb8 FGA cls + regression head
    flatten = mx.symbol.flatten(data=roi_pool, name="rcnn_bb8FGA_flatten")
    fc6_weight = mx.symbol.Variable(name="rcnn_bb8FGA_fc6_weight", init=mx.init.Xavier())
    fc6_bias = mx.symbol.Variable(name="rcnn_bb8FGA_fc6_bias", init=mx.init.Constant(0.0))
    fc6 = mx.symbol.FullyConnected(data=flatten, weight=fc6_weight, bias=fc6_bias, num_hidden=1024,
                                   name="rcnn_bb8FGA_fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_bb8FGA_relu6")
    fc7_weight = mx.symbol.Variable(name="rcnn_bb8FGA_fc7_weight", init=mx.init.Xavier())
    fc7_bias = mx.symbol.Variable(name="rcnn_bb8FGA_fc7_bias", init=mx.init.Constant(0.0))
    fc7 = mx.symbol.FullyConnected(data=relu6, weight=fc7_weight, bias=fc7_bias, num_hidden=1024,
                                   name="rcnn_bb8FGA_fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_bb8FGA_relu7")

    # FGA cls
    cls_pred_weight = mx.symbol.Variable(name="rcnn_bb8FGA_cls_pred_weight", init=mx.init.Normal(sigma=0.01))
    cls_pred_bias = mx.symbol.Variable(name="rcnn_bb8FGA_cls_pred_bias", init=mx.init.Constant(0.0))
    rcnn_bb8FGA_cls_score = mx.symbol.FullyConnected(data=relu7, weight=cls_pred_weight, bias=cls_pred_bias,
                                                     num_hidden=8 * granularity[0] * granularity[1], name="rcnn_bb8FGA_cls_score")
    # FGA offset reg
    reg_pred_weight = mx.symbol.Variable(name="rcnn_bb8FGA_reg_pred_weight", init=mx.init.Normal(sigma=0.001))
    reg_pred_bias = mx.symbol.Variable(name="rcnn_bb8FGA_reg_pred_bias", init=mx.init.Constant(0.0))
    rcnn_bb8FGA_reg_pred = mx.symbol.FullyConnected(data=relu7, weight=reg_pred_weight, bias=reg_pred_bias,
                                                       num_hidden=16, name="rcnn_bb8FGA_reg_pred")

    # # FGA cls loss
    rcnn_bb8FGA_cls_prob = mx.symbol.SoftmaxOutput(data=rcnn_bb8FGA_cls_score.reshape((0,8,-1)), label=FGA_cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=alpha_bb8, preserve_shape=True, \
        normalization='valid', name="rcnn_bb8FGA_cls_prob")
    # FGA offset reg loss
    rcnn_bb8FGA_reg_loss_ = mx.symbol.smooth_l1(name="rcnn_bb8FGA_reg_loss_",
                                                   data=FGA_reg_weight * (
                                                               rcnn_bb8FGA_reg_pred - FGA_reg_target),
                                                   scalar=1.0)
    rcnn_bb8FGA_reg_loss = mx.symbol.MakeLoss(rcnn_bb8FGA_reg_loss_, grad_scale=alpha_bb8, \
                                                 normalization='valid', name="rcnn_bb8FGA_reg_loss")

    # heatmap L2 loss
    # rcnn_FGA_cls_loss_ = mx.symbol.square(data=(rcnn_FGA_cls_score - FGA_cls_target), name='rcnn_FGA_cls_loss_') / 2.
    # rcnn_FGA_cls_loss = mx.symbol.MakeLoss(name='rcnn_FGA_cls_loss', data=rcnn_FGA_cls_loss_, grad_scale=0.0,
    #                                         normalization='batch')


    # rcnn_FGA_bb8_reg_loss_ = FGA_reg_weight * mx.symbol.smooth_l1(name='rcnn_FGA_bb8_reg_loss_', scalar=1.0,
    #                                                data=(rcnn_FGA_bb8_pred.reshape((0, 16, granularity[0], granularity[1])) - FGA_reg_target))

    # rcnn_FGA_bb8_reg_loss_ = FGA_reg_weight * mx.symbol.smooth_l1(name='rcnn_FGA_bb8_reg_loss_', scalar=1.0,
    #                                                               data=(rcnn_FGA_bb8_pred - FGA_reg_target))
    #
    # rcnn_FGA_bb8_reg_loss = mx.sym.MakeLoss(name='rcnn_FGA_bb8_reg_loss', data=rcnn_FGA_bb8_reg_loss_, grad_scale=1.0,
    #                                         normalization='batch', valid_thresh=1e-12)
    # rcnn_group = [rcnn_FGA_cls_loss, rcnn_FGA_bb8_reg_loss]

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    rpn_loc_target = mx.symbol.MakeLoss(data=loc_target * loc_target_mask, grad_scale=0, name="rpn_loc_target")
    det_out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
                                                  name="rpn_det_out", nms_threshold=0.45, force_suppress=False,
                                                  variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
    det = mx.symbol.MakeLoss(data=det_out, grad_scale=0, name="det_out")
    # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0., name="roi_pool_monitor")

    rois = mx.symbol.MakeLoss(data=rois, grad_scale=0, name="rois")
    score = mx.symbol.MakeLoss(data=score, grad_scale=0, name="score")
    cid = mx.symbol.MakeLoss(data=cid, grad_scale=0, name="cid")
    #
    rcnn_FGA_cls_target = mx.symbol.MakeLoss(data=FGA_cls_target, grad_scale=0, name="rcnn_FGA_cls_target")
    rcnn_FGA_reg_target = mx.symbol.MakeLoss(data=FGA_reg_target, grad_scale=0, name="rcnn_FGA_reg_target")
    # rcnn_FGA_cls_score = mx.symbol.MakeLoss(data=rcnn_FGA_cls_score, grad_scale=0, name="rcnn_FGA_cls_score_monitor")
    # rcnn_FGA_bb8_pred = mx.symbol.MakeLoss(data=rcnn_FGA_bb8_pred, grad_scale=0, name="rcnn_FGA_bb8_pred_monitor")

    # group output
    # out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
    #                        anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    # out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, rpn_loc_target, det,
                           rois, score, cid,
                           rcnn_FGA_cls_target, rcnn_FGA_reg_target, rcnn_bb8FGA_cls_prob, rcnn_bb8FGA_reg_loss
                           ])
    return out

def get_FGARCNN_cls_softmax_reg_offset_resnetm_fpn_test(num_classes, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], im_info=(), **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    conv_fpn_feat_dict, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    # rpn_bbox_pred : (N, 4 x num_anchors, H, W)
    # rpn_cls_pred : (N, num_anchor x (num_classes + 1), H, W)
    # rpn_anchor: (N, num_all_anchor, 4)
    rpn_bbox_pred_dict, rpn_cls_prob_dict, rpn_anchor_dict, \
    loc_preds, cls_preds, anchor_boxes = multibox_layer_FPN_RCNN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    cls_prob = mx.symbol.softmax(data=cls_preds, axis=1, name="rpn_cls_prob")

    # rpn proposal
    # im_info = (512, 512, 1)    # (H, W, scale)
    rpn_feat_stride = [4, 8, 16, 32, 64]
    granularity = (5, 5)

    # rpn detection results merging all the levels, set a higher nms threshold to keep more proposals
    rpn_det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="rpn_proposal", nms_threshold=0.45, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)

    # # select foreground region proposals, and transform the coordinate from [0,1] to [0, 448]
    rois, score, cid = mx.symbol.Custom(op_type='rpn_proposal',
                     rpn_det=rpn_det,
                     output_score=True,
                     rpn_post_nms_top_n=400, im_info=im_info
                     )
    rois = mx.symbol.reshape(rois, shape=(-1, 5))

    # # rcnn roi pool
    roi_pool = mx.symbol.contrib.ROIAlign(
        name='roi_pool', data=conv_fpn_feat_dict['stride8'], rois=rois, pooled_size=(7, 7),
        spatial_scale=1.0 / 8.)

    # bb8 FGA cls + regression head
    flatten = mx.symbol.flatten(data=roi_pool, name="rcnn_bb8FGA_flatten")
    fc6_weight = mx.symbol.Variable(name="rcnn_bb8FGA_fc6_weight", init=mx.init.Xavier())
    fc6_bias = mx.symbol.Variable(name="rcnn_bb8FGA_fc6_bias", init=mx.init.Constant(0.0))
    fc6 = mx.symbol.FullyConnected(data=flatten, weight=fc6_weight, bias=fc6_bias, num_hidden=1024,
                                   name="rcnn_bb8FGA_fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_bb8FGA_relu6")
    fc7_weight = mx.symbol.Variable(name="rcnn_bb8FGA_fc7_weight", init=mx.init.Xavier())
    fc7_bias = mx.symbol.Variable(name="rcnn_bb8FGA_fc7_bias", init=mx.init.Constant(0.0))
    fc7 = mx.symbol.FullyConnected(data=relu6, weight=fc7_weight, bias=fc7_bias, num_hidden=1024,
                                   name="rcnn_bb8FGA_fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_bb8FGA_relu7")

    # FGA cls
    cls_pred_weight = mx.symbol.Variable(name="rcnn_bb8FGA_cls_pred_weight", init=mx.init.Normal(sigma=0.01))
    cls_pred_bias = mx.symbol.Variable(name="rcnn_bb8FGA_cls_pred_bias", init=mx.init.Constant(0.0))
    rcnn_bb8FGA_cls_score = mx.symbol.FullyConnected(data=relu7, weight=cls_pred_weight, bias=cls_pred_bias,
                                                     num_hidden=8 * granularity[0] * granularity[1], name="rcnn_bb8FGA_cls_score")
    rcnn_bb8FGA_cls_score_reshape = mx.symbol.reshape(rcnn_bb8FGA_cls_score, shape=(0, 8, granularity[0], granularity[1]))
    # FGA offset reg
    reg_pred_weight = mx.symbol.Variable(name="rcnn_bb8FGA_reg_pred_weight", init=mx.init.Normal(sigma=0.001))
    reg_pred_bias = mx.symbol.Variable(name="rcnn_bb8FGA_reg_pred_bias", init=mx.init.Constant(0.0))
    rcnn_bb8FGA_reg_pred = mx.symbol.FullyConnected(data=relu7, weight=reg_pred_weight, bias=reg_pred_bias,
                                                       num_hidden=16, name="rcnn_bb8FGA_reg_pred")

    out = mx.symbol.Group([rois, score, cid, rcnn_bb8FGA_cls_score_reshape, rcnn_bb8FGA_reg_pred])
    return out

def get_RCNN_boundary_offset_resnetm_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], im_info=(), **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down, pose_module
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    conv_fpn_feat_dict, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    # rpn_bbox_pred : (N, 4 x num_anchors, H, W)
    # rpn_cls_pred : (N, num_anchor x (num_classes + 1), H, W)
    # rpn_anchor: (N, num_all_anchor, 4)
    rpn_bbox_pred_dict, rpn_cls_prob_dict, rpn_anchor_dict, \
    loc_preds, cls_preds, anchor_boxes = multibox_layer_FPN_RCNN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    rpn_targets = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.4, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = rpn_targets[0]
    loc_target_mask = rpn_targets[1]
    cls_target = rpn_targets[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1.0, multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1.0, \
        normalization='valid', name="loc_loss")

    # rpn detection results merging all the levels, set a higher nms threshold to keep more proposals
    rpn_det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="rpn_proposal", nms_threshold=0.7, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)

    # # select foreground region proposals, and transform the coordinate from [0,1] to [0, 448]
    rois, score, cid = mx.symbol.Custom(op_type='rpn_proposal',
                     rpn_det=rpn_det,
                     output_score=True,
                     rpn_post_nms_top_n=400, im_info=im_info
                     )
    rois = mx.symbol.reshape(rois, shape=(-1, 5))
    # rois = mx.symbol.MakeLoss(data=rois, grad_scale=0, name='rpn_roi')
    # score = mx.symbol.MakeLoss(data=score, grad_scale=0, name='rpn_score')
    # cid = mx.symbol.MakeLoss(data=cid, grad_scale=0, name='rpn_cid')
    #     rpn_proposal_dict.update({'rpn_proposal_stride%s' % s: rpn_det})

    # rcnn roi proposal target
    group = mx.symbol.Custom(rois=rois, gt_boxes=label, op_type='bb8_proposal_target_boundary_offset_soft_cls',
                         num_keypoints=8, batch_images=4,
                         batch_rois=512, fg_fraction=1.0,
                         fg_overlap=0.5, bb8_variance=(0.1, 0.1),
                         im_info=im_info)
    rois = group[0]
    boundary_cls_target = group[1]    # (N, 8, 4) for soft cls, (N, 8) for hard cls
    boundary_reg_target = group[2]    # (N, 16)
    boundary_reg_weight = group[3]

    # # rcnn roi pool
    # conv_feat_kp = pose_module(conv_fpn_feat_dict['stride4'], conv_fpn_feat_dict['stride8'], conv_fpn_feat_dict['stride16'])
    roi_pool = mx.symbol.contrib.ROIAlign(
        name='roi_pool', data=conv_fpn_feat_dict['stride4'], rois=rois, pooled_size=(7, 7),
        spatial_scale=1.0 / 4.)
    
    # roi_pool = mx.symbol.Custom(op_type="fpn_roi_pool",
    #                             rcnn_strides="(16,8,4)",
    #                             pool_h=14, pool_w=14,
    #                             feat_stride16=conv_fpn_feat_dict['stride16'],
    #                             feat_stride8=conv_fpn_feat_dict['stride8'],
    #                             feat_stride4=conv_fpn_feat_dict['stride4'],
    #                             rois=rois)
    # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0., name='roi_pool')

    # bb8 boundary cls + regression head
    flatten = mx.symbol.flatten(data=roi_pool, name="rcnn_bb8boundary_flatten")
    fc6_weight = mx.symbol.Variable(name="rcnn_bb8boundary_fc6_weight", init=mx.init.Xavier())
    fc6_bias = mx.symbol.Variable(name="rcnn_bb8boundary_fc6_bias", init=mx.init.Constant(0.0))
    fc6 = mx.symbol.FullyConnected(data=flatten, weight=fc6_weight, bias=fc6_bias, num_hidden=1024,
                                   name="rcnn_bb8boundary_fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_bb8boundary_relu6")
    fc7_weight = mx.symbol.Variable(name="rcnn_bb8boundary_fc7_weight", init=mx.init.Xavier())
    fc7_bias = mx.symbol.Variable(name="rcnn_bb8boundary_fc7_bias", init=mx.init.Constant(0.0))
    fc7 = mx.symbol.FullyConnected(data=relu6, weight=fc7_weight, bias=fc7_bias, num_hidden=1024,
                                   name="rcnn_bb8boundary_fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_bb8boundary_relu7")

    # boundary cls
    cls_pred_weight = mx.symbol.Variable(name="rcnn_bb8boundary_cls_pred_weight", init=mx.init.Normal(sigma=0.01))
    cls_pred_bias = mx.symbol.Variable(name="rcnn_bb8boundary_cls_pred_bias", init=mx.init.Constant(0.0))
    rcnn_bb8boundary_cls_score = mx.symbol.FullyConnected(data=relu7, weight=cls_pred_weight, bias=cls_pred_bias,
                                                     num_hidden=8 * 4, name="rcnn_bb8boundary_cls_score")
    # boundary offset reg
    reg_pred_weight = mx.symbol.Variable(name="rcnn_bb8boundary_reg_pred_weight", init=mx.init.Normal(sigma=0.001))
    reg_pred_bias = mx.symbol.Variable(name="rcnn_bb8boundary_reg_pred_bias", init=mx.init.Constant(0.0))
    rcnn_bb8boundary_reg_pred = mx.symbol.FullyConnected(data=relu7, weight=reg_pred_weight, bias=reg_pred_bias,
                                                       num_hidden=8 * 4 * 2, name="rcnn_bb8boundary_reg_pred")

    # # cls loss
    # rcnn_bb8boundary_cls_prob = mx.symbol.SoftmaxOutput(data=rcnn_bb8boundary_cls_score.reshape((0,8,-1)),
    #                                                     label=boundary_cls_target, \
    #     ignore_label=-1, use_ignore=True, grad_scale=alpha_bb8, preserve_shape=True, \
    #     normalization='valid', name="rcnn_bb8boundary_cls_prob")
    rcnn_bb8boundary_reg_loss_ = mx.symbol.smooth_l1(name="rcnn_bb8boundary_reg_loss_",
                                                     data=boundary_reg_weight * (
                                                             rcnn_bb8boundary_reg_pred - boundary_reg_target),
                                                     scalar=1.0)
    rcnn_bb8boundary_cls_prob = mx.symbol.Custom(cls_score=rcnn_bb8boundary_cls_score.reshape((0, 8, -1)), label=boundary_cls_target,
                                                 op_type="softcrossentropyloss",
                                                 ignore_label=-1, normalization='valid', grad_scale=alpha_bb8)
    # offset reg loss
    # no_grad_cls_prob = mx.symbol.pick(rcnn_bb8boundary_cls_prob, boundary_cls_target, axis=-1, keepdims=True)    # shape (N, 8, 1)
    # no_grad_cls_prob = mx.symbol.pow(no_grad_cls_prob, 2)
    # no_grad_cls_prob = mx.symbol.broadcast_div(no_grad_cls_prob, mx.symbol.max(no_grad_cls_prob))
    # no_grad_cls_prob = mx.symbol.BlockGrad(1.1 * no_grad_cls_prob)
    # rcnn_bb8boundary_reg_loss_ = mx.symbol.reshape(rcnn_bb8boundary_reg_loss_, shape=(0, 8, -1))
    # rcnn_bb8boundary_reg_loss_ = mx.symbol.broadcast_mul(rcnn_bb8boundary_reg_loss_, no_grad_cls_prob)
    rcnn_bb8boundary_reg_loss = mx.symbol.MakeLoss(rcnn_bb8boundary_reg_loss_, grad_scale=alpha_bb8, \
                                                 normalization='valid', name="rcnn_bb8boundary_reg_loss")

    # heatmap L2 loss
    # rcnn_FGA_cls_loss_ = mx.symbol.square(data=(rcnn_FGA_cls_score - FGA_cls_target), name='rcnn_FGA_cls_loss_') / 2.
    # rcnn_FGA_cls_loss = mx.symbol.MakeLoss(name='rcnn_FGA_cls_loss', data=rcnn_FGA_cls_loss_, grad_scale=0.0,
    #                                         normalization='batch')


    # rcnn_FGA_bb8_reg_loss_ = FGA_reg_weight * mx.symbol.smooth_l1(name='rcnn_FGA_bb8_reg_loss_', scalar=1.0,
    #                                                data=(rcnn_FGA_bb8_pred.reshape((0, 16, granularity[0], granularity[1])) - FGA_reg_target))

    # rcnn_FGA_bb8_reg_loss_ = FGA_reg_weight * mx.symbol.smooth_l1(name='rcnn_FGA_bb8_reg_loss_', scalar=1.0,
    #                                                               data=(rcnn_FGA_bb8_pred - FGA_reg_target))
    #
    # rcnn_FGA_bb8_reg_loss = mx.sym.MakeLoss(name='rcnn_FGA_bb8_reg_loss', data=rcnn_FGA_bb8_reg_loss_, grad_scale=1.0,
    #                                         normalization='batch', valid_thresh=1e-12)
    # rcnn_group = [rcnn_FGA_cls_loss, rcnn_FGA_bb8_reg_loss]

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    rpn_loc_target = mx.symbol.MakeLoss(data=loc_target * loc_target_mask, grad_scale=0, name="rpn_loc_target")
    det_out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
                                                  name="rpn_det_out", nms_threshold=0.45, force_suppress=False,
                                                  variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
    det = mx.symbol.MakeLoss(data=det_out, grad_scale=0, name="det_out")
    # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0., name="roi_pool_monitor")

    rois = mx.symbol.MakeLoss(data=rois, grad_scale=0, name="rois")
    score = mx.symbol.MakeLoss(data=score, grad_scale=0, name="score")
    cid = mx.symbol.MakeLoss(data=cid, grad_scale=0, name="cid")
    #
    rcnn_boundary_cls_target = mx.symbol.MakeLoss(data=boundary_cls_target, grad_scale=0, name="rcnn_boundary_cls_target")
    rcnn_boundary_reg_target = mx.symbol.MakeLoss(data=boundary_reg_target, grad_scale=0, name="rcnn_boundary_reg_target")
    # rcnn_FGA_cls_score = mx.symbol.MakeLoss(data=rcnn_FGA_cls_score, grad_scale=0, name="rcnn_FGA_cls_score_monitor")
    # rcnn_FGA_bb8_pred = mx.symbol.MakeLoss(data=rcnn_FGA_bb8_pred, grad_scale=0, name="rcnn_FGA_bb8_pred_monitor")

    # group output
    # out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
    #                        anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    # out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, rpn_loc_target, det,
                           rois, score, cid,
                           rcnn_boundary_cls_target, rcnn_boundary_reg_target, rcnn_bb8boundary_cls_prob, rcnn_bb8boundary_reg_loss
                           ])
    return out

def get_RCNN_boundary_offset_resnetm_fpn_test(num_classes, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], im_info=(), **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down, pose_module
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    conv_fpn_feat_dict, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    # rpn_bbox_pred : (N, 4 x num_anchors, H, W)
    # rpn_cls_pred : (N, num_anchor x (num_classes + 1), H, W)
    # rpn_anchor: (N, num_all_anchor, 4)
    rpn_bbox_pred_dict, rpn_cls_prob_dict, rpn_anchor_dict, \
    loc_preds, cls_preds, anchor_boxes = multibox_layer_FPN_RCNN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    # now cls_preds are in shape of  batchsize x num_class x num_anchors
    cls_prob = mx.symbol.softmax(data=cls_preds, axis=1, name="rpn_cls_prob")

    # rpn detection results merging all the levels, set a higher nms threshold to keep more proposals
    rpn_det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="rpn_proposal", nms_threshold=0.45, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)

    # # select foreground region proposals, and transform the coordinate from [0,1] to [0, 448]
    rois, score, cid = mx.symbol.Custom(op_type='rpn_proposal',
                     rpn_det=rpn_det,
                     output_score=True,
                     rpn_post_nms_top_n=400, im_info=im_info
                     )
    rois = mx.symbol.reshape(rois, shape=(-1, 5))

    # rcnn roi pool
    # conv_feat_kp = pose_module(conv_fpn_feat_dict['stride4'], conv_fpn_feat_dict['stride8'],
    #                            conv_fpn_feat_dict['stride16'])
    roi_pool = mx.symbol.contrib.ROIAlign(
        name='roi_pool', data=conv_fpn_feat_dict['stride4'], rois=rois, pooled_size=(7, 7),
        spatial_scale=1.0 / 4.)

    # bb8 boundary cls + regression head
    flatten = mx.symbol.flatten(data=roi_pool, name="rcnn_bb8boundary_flatten")
    fc6_weight = mx.symbol.Variable(name="rcnn_bb8boundary_fc6_weight", init=mx.init.Xavier())
    fc6_bias = mx.symbol.Variable(name="rcnn_bb8boundary_fc6_bias", init=mx.init.Constant(0.0))
    fc6 = mx.symbol.FullyConnected(data=flatten, weight=fc6_weight, bias=fc6_bias, num_hidden=1024,
                                   name="rcnn_bb8boundary_fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_bb8boundary_relu6")
    fc7_weight = mx.symbol.Variable(name="rcnn_bb8boundary_fc7_weight", init=mx.init.Xavier())
    fc7_bias = mx.symbol.Variable(name="rcnn_bb8boundary_fc7_bias", init=mx.init.Constant(0.0))
    fc7 = mx.symbol.FullyConnected(data=relu6, weight=fc7_weight, bias=fc7_bias, num_hidden=1024,
                                   name="rcnn_bb8boundary_fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_bb8boundary_relu7")

    # boundary cls
    cls_pred_weight = mx.symbol.Variable(name="rcnn_bb8boundary_cls_pred_weight", init=mx.init.Normal(sigma=0.01))
    cls_pred_bias = mx.symbol.Variable(name="rcnn_bb8boundary_cls_pred_bias", init=mx.init.Constant(0.0))
    rcnn_bb8boundary_cls_score = mx.symbol.FullyConnected(data=relu7, weight=cls_pred_weight, bias=cls_pred_bias,
                                                     num_hidden=8 * 4, name="rcnn_bb8boundary_cls_score")
    # boundary offset reg
    reg_pred_weight = mx.symbol.Variable(name="rcnn_bb8boundary_reg_pred_weight", init=mx.init.Normal(sigma=0.001))
    reg_pred_bias = mx.symbol.Variable(name="rcnn_bb8boundary_reg_pred_bias", init=mx.init.Constant(0.0))
    rcnn_bb8boundary_reg_pred = mx.symbol.FullyConnected(data=relu7, weight=reg_pred_weight, bias=reg_pred_bias,
                                                       num_hidden=8 * 4 * 2, name="rcnn_bb8boundary_reg_pred")
    rcnn_bb8boundary_cls_score_reshape = mx.symbol.reshape(rcnn_bb8boundary_cls_score, shape=(0, 8, 4))

    out = mx.symbol.Group([rois, score, cid, rcnn_bb8boundary_cls_score_reshape, rcnn_bb8boundary_reg_pred])
    return out


def get_RCNN_resnetm_fpn_test(num_classes, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    from rcnn.config import config
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    conv_fpn_feat_dict, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    # rpn_bbox_pred : (N, 4 x num_anchors, H, W)
    # rpn_cls_pred : (N, num_anchor x (num_classes + 1), H, W)
    # rpn_anchor: (N, num_all_anchor, 4)
    rpn_bbox_pred_dict, rpn_cls_prob_dict, rpn_anchor_dict, \
    loc_preds, cls_preds, anchor_boxes = multibox_layer_FPN_RCNN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.softmax(data=cls_preds, axis=1, name="rpn_cls_prob")

    # rpn proposal
    im_info = (512, 512, 1)    # (H, W, scale)
    feat_stride = [4, 8, 16, 32, 64]
    granularity = (56, 56)

    rpn_det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="rpn_proposal", nms_threshold=0.45, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)

    rois, score, cid = mx.symbol.Custom(op_type='rpn_proposal',
                     rpn_det=rpn_det,
                     output_score=True,
                     rpn_post_nms_top_n=400, im_info=im_info
                     )

    rois = mx.symbol.reshape(rois, shape=(-1, 5))

    # rcnn roi pool
    roi_pool = mx.symbol.Custom(op_type="fpn_roi_pool",
                                rcnn_strides="(16,8,4)",
                                pool_h=14, pool_w=14,
                                feat_stride16=conv_fpn_feat_dict['stride16'],
                                feat_stride8=conv_fpn_feat_dict['stride8'],
                                feat_stride4=conv_fpn_feat_dict['stride4'],
                                rois=rois)
    # roi_pool = mx.symbol.MakeLoss(data=roi_pool, grad_scale=0., name='roi_pool')

    # # Mask RCNN head
    for i in range(8):
        weight = mx.symbol.Variable(name="rcnn_FGA_cls_conv{}_weight".format(i + 1))
        bias = mx.symbol.Variable(name="rcnn_FGA_cls_conv{}_bias".format(i + 1))
        x = mx.symbol.Convolution(data=roi_pool, weight=weight, bias=bias,
                                  kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=512,
                                  name="rcnn_FGA_cls_conv{}".format(i + 1))
        x = mx.symbol.Activation(data=x, act_type="relu", name="rcnn_FGA_cls_relu{}".format(i + 1))

    deconv_weight = mx.symbol.Variable(name="rcnn_FGA_cls_deconv_weight")
    deconv_bias = mx.symbol.Variable(name="rcnn_FGA_cls_deconv_bias")
    x = mx.symbol.Deconvolution(data=x, weight=deconv_weight, bias=deconv_bias,
                                kernel=(4, 4), stride=(2, 2), pad=(1, 1), target_shape=(28, 28), num_filter=8,
                                no_bias=False, name="rcnn_FGA_cls_deconv")

    rcnn_FGA_cls_score = mx.symbol.contrib.BilinearResize2D(data=x, height=granularity[0], width=granularity[1], name="rcnn_FGA_cls_score")

    out = mx.symbol.Group([rois, score, cid, rcnn_FGA_cls_score])
    return out


def get_resnetmd_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_md_conv, get_ssd_md_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_md_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_ssd_md_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_FPN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_vgg_reduced_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.vgg16_reduced import get_vgg_reduced_conv, get_vgg_reduced_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_vgg_reduced_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_vgg_reduced_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_symbol(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    body = import_module(network).get_symbol(num_classes=num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return out
