import mxnet as mx
import numpy as np
from MultiBoxDetection import RCNNFGABB8MultiBoxDetection

class MultiBoxMetric_RCNN_offset(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(MultiBoxMetric_RCNN_offset, self).__init__('MultiBox_bb8offset')
        self.eps = eps
        self.num = 3
        self.name = ['rpn_CrossEntropy', 'rpn_SmoothL1',
                     'rcnn_bb8offset_reg_SmoothL1']
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        labels = labels[0].asnumpy()
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()
        rpn_loc_target = preds[3].asnumpy()
        # rpn_loc_target_valid = rpn_loc_target[np.nonzero(rpn_loc_target)]
        # rpn_loc_target_valid_abs_mean = np.mean(np.abs(rpn_loc_target_valid))
        # rpn_loc_target_valid_mean = np.mean(rpn_loc_target_valid)
        # rpn_loc_target_valid_variance = np.var(rpn_loc_target_valid)
        dets = preds[4].asnumpy()
        dets_valid_index = np.where(dets[:,:, 0] >= 0)
        dets_valid = dets[dets_valid_index[0], dets_valid_index[1]]

        # ssd as rpn loss count
        valid_count = np.sum(cls_label >= 0)
        box_count = np.sum(cls_label > 0)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        # in case you have a 'other' class
        label[np.where(label >= cls_prob.shape[1])] = 0
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]
        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count
        # smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += box_count * 4

        # rpn proposals
        # rpn_rois = preds[5].asnumpy()
        # rpn_score = preds[6].asnumpy()
        # rpn_cid = preds[7].asnumpy()

        # rcnn_FGA_cls_loss = preds[4].asnumpy()
        # rcnn_FGA_bb8_reg_loss = preds[5].asnumpy()

        # rcnn FGA loss count
        rcnn_bb8offset_reg_target = preds[8].asnumpy()    # shape (N_rois, 2*num_keypoints)
        rcnn_bb8offset_reg_loss = preds[9].asnumpy()    # shape (N_rois, 2*num_keypoints)

        # rcnn_FGA_reg_target_abs_mean = np.mean(np.abs(rcnn_FGA_reg_target))
        # rcnn_FGA_reg_target_mean = np.mean(rcnn_FGA_reg_target)
        # rcnn_FGA_reg_target_variance = np.var(rcnn_FGA_reg_target)

        # bb8offset reg loss update
        rcnn_valid_count = len(np.nonzero(rcnn_bb8offset_reg_target)[0])
        self.sum_metric[2] += np.sum(rcnn_bb8offset_reg_loss)
        self.num_inst[2] += rcnn_valid_count

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

class MultiBoxMetric_MaskRCNN_keypoint(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(MultiBoxMetric_MaskRCNN_keypoint, self).__init__('MultiBox_softmax')
        self.eps = eps
        self.num = 4
        self.name = ['rpn_CrossEntropy', 'rpn_SmoothL1', 'maskrcnn_keypoint_cls_CrossEntropy', 'maskrcnn_keypoint_cls_accuracy']
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        labels = labels[0].asnumpy()
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()

        # ssd as rpn loss count
        valid_count = np.sum(cls_label >= 0)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        # in case you have a 'other' class
        label[np.where(label >= cls_prob.shape[1])] = 0
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]
        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count
        # smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += valid_count

        # maskrcnn keypoint loss count
        maskrcnn_keypoint_cls_target = preds[4].asnumpy()    # shape (N_rois, num_keypoints)
        maskrcnn_keypoint_cls_prob = preds[5].asnumpy()     # shape (N_rois, num_keypoints, granularity[0]*granularity[1])

        # softmax version loss update
        rcnn_valid_count = np.sum(maskrcnn_keypoint_cls_target >= 0)
        rcnn_FGA_cls_target = maskrcnn_keypoint_cls_target.flatten()
        rcnn_mask = np.where(rcnn_FGA_cls_target >= 0)[0]
        rcnn_indices = np.int64(rcnn_FGA_cls_target[rcnn_mask])
        rcnn_prob = maskrcnn_keypoint_cls_prob.reshape((-1, maskrcnn_keypoint_cls_prob.shape[2]))
        rcnn_prob = rcnn_prob[rcnn_mask, rcnn_indices]
        self.sum_metric[2] += (-np.log(rcnn_prob +self.eps)).sum()
        self.num_inst[2] += rcnn_valid_count

        max_pred_prob_indices = np.argmax(maskrcnn_keypoint_cls_prob.reshape((-1, maskrcnn_keypoint_cls_prob.shape[2])), axis=-1)
        max_pred_prob_indices = max_pred_prob_indices[rcnn_mask]
        accuracy = np.sum(max_pred_prob_indices == rcnn_indices)
        self.sum_metric[3] += np.sum(accuracy)
        self.num_inst[3] += rcnn_valid_count

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

class MultiBoxMetric_softmax(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(MultiBoxMetric_softmax, self).__init__('MultiBox_softmax')
        self.eps = eps
        self.num = 6
        self.name = ['rpn_CrossEntropy', 'rpn_SmoothL1', 'rcnn_FGA_cls_CrossEntropy', 'rcnn_FGA_cls_accuracy',
                     'rcnn_FGA_bb8_pred_SmoothL1', 'rcnn_FGA_reg_mae_pixel' ]
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        labels = labels[0].asnumpy()
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()
        rpn_loc_target = preds[3].asnumpy()
        # rpn_loc_target_valid = rpn_loc_target[np.nonzero(rpn_loc_target)]
        # rpn_loc_target_valid_abs_mean = np.mean(np.abs(rpn_loc_target_valid))
        # rpn_loc_target_valid_mean = np.mean(rpn_loc_target_valid)
        # rpn_loc_target_valid_variance = np.var(rpn_loc_target_valid)
        dets = preds[4].asnumpy()
        dets_valid_index = np.where(dets[:,:, 0] >= 0)
        dets_valid = dets[dets_valid_index[0], dets_valid_index[1]]

        # ssd as rpn loss count
        valid_count = np.sum(cls_label >= 0)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        # in case you have a 'other' class
        label[np.where(label >= cls_prob.shape[1])] = 0
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]
        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count
        # smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += valid_count

        # rpn proposals
        # rpn_rois = preds[5].asnumpy()
        # rpn_score = preds[6].asnumpy()
        # rpn_cid = preds[7].asnumpy()

        # rcnn_FGA_cls_loss = preds[4].asnumpy()
        # rcnn_FGA_bb8_reg_loss = preds[5].asnumpy()

        # rcnn FGA loss count
        rcnn_FGA_cls_target = preds[8].asnumpy()    # shape (N_rois, num_keypoints)
        rcnn_FGA_reg_target = preds[9].asnumpy()    # shape (N_rois, 2*num_keypoints)
        rcnn_FGA_cls_prob = preds[10].asnumpy()     # shape (N_rois, num_keypoints, granularity[0]*granularity[1])

        # rcnn_FGA_reg_target_abs_mean = np.mean(np.abs(rcnn_FGA_reg_target))
        # rcnn_FGA_reg_target_mean = np.mean(rcnn_FGA_reg_target)
        # rcnn_FGA_reg_target_variance = np.var(rcnn_FGA_reg_target)


        # softmax version loss update
        rcnn_valid_count = np.sum(rcnn_FGA_cls_target >= 0)
        rcnn_FGA_cls_target = rcnn_FGA_cls_target.flatten()
        rcnn_mask = np.where(rcnn_FGA_cls_target >= 0)[0]
        rcnn_indices = np.int64(rcnn_FGA_cls_target[rcnn_mask])
        rcnn_prob = rcnn_FGA_cls_prob.reshape((-1, rcnn_FGA_cls_prob.shape[2]))
        rcnn_prob = rcnn_prob[rcnn_mask, rcnn_indices]
        self.sum_metric[2] += (-np.log(rcnn_prob +self.eps)).sum()
        self.num_inst[2] += rcnn_valid_count

        max_pred_prob_indices = np.argmax(rcnn_FGA_cls_prob.reshape((-1, rcnn_FGA_cls_prob.shape[2])), axis=-1)
        max_pred_prob_indices = max_pred_prob_indices[rcnn_mask]
        accuracy = np.sum(max_pred_prob_indices == rcnn_indices)
        self.sum_metric[3] += np.sum(accuracy)
        self.num_inst[3] += rcnn_valid_count

        # heatmap version loss update
        # logistic_loss = - rcnn_FGA_cls_target * np.log(rcnn_FGA_cls_prob) - (1 - rcnn_FGA_cls_target) * np.log(1 - rcnn_FGA_cls_prob)

        # self.sum_metric[2] += np.mean(rcnn_FGA_cls_loss)
        # self.num_inst[2] += 1

        # rcnn_FGA_bb8_reg_loss = np.sum(rcnn_FGA_bb8_reg_loss, axis=(2,3))
        # self.sum_metric[3] += np.mean(rcnn_FGA_bb8_reg_loss)
        # self.num_inst[3] += 1

        # rcnn_FGA_cls_score = preds[12].asnumpy()
        # rcnn_FGA_bb8_pred = preds[13].asnumpy()

        # rpn_rois_width = rpn_rois[:, 3] - rpn_rois[:, 1]
        # rpn_rois_height = rpn_rois[:, 4] - rpn_rois[:, 2]
        # rcnn_FGA_reg_pred_error = np.abs((rcnn_FGA_bb8_pred - rcnn_FGA_reg_target) * 0.1).reshape(rpn_rois.shape[0], 8, 2)
        # rcnn_FGA_reg_pred_error_x = rcnn_FGA_reg_pred_error[:, :, 0] * rpn_rois_width[:, np.newaxis]
        # rcnn_FGA_reg_pred_error_y = rcnn_FGA_reg_pred_error[:, :, 1] * rpn_rois_height[:, np.newaxis]
        # rcnn_FGA_reg_mae_pixel = np.concatenate((rcnn_FGA_reg_pred_error_x[:, :, np.newaxis],
        #                                          rcnn_FGA_reg_pred_error_y[:, :, np.newaxis]), axis=2)
        # self.sum_metric[4] += np.sum(rcnn_FGA_reg_mae_pixel)
        # self.num_inst[4] += rcnn_FGA_reg_mae_pixel.size

        # rcnn_FGA_cls_target_max_index = np.argmax(rcnn_FGA_cls_target.reshape(rcnn_FGA_cls_target.shape[0],
        #                                                                       rcnn_FGA_cls_target.shape[1], -1)
        #                                         , axis=-1)
        # rcnn_FGA_cls_score_max_index = np.argmax(rcnn_FGA_cls_score.reshape(rcnn_FGA_cls_score.shape[0],
        #                                                                     rcnn_FGA_cls_score.shape[1], -1),
        #                                          axis=-1)
        # index_correct = rcnn_FGA_cls_score_max_index == rcnn_FGA_cls_target_max_index
        # self.sum_metric[5] += np.sum(index_correct)
        # self.num_inst[5] += rcnn_FGA_cls_target_max_index.size

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)


class MultiBoxMetric_heatmap(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """

    def __init__(self, eps=1e-8):
        super(MultiBoxMetric_heatmap, self).__init__('MultiBox')
        self.eps = eps
        self.num = 6
        self.name = ['CrossEntropy', 'SmoothL1', 'rcnn_FGA_cls_L2Loss', 'rcnn_FGA_bb8_pred_SmoothL1',
                     'rcnn_FGA_reg_mae_pixel', 'rcnn_FGA_cls_accuracy']
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        labels = labels[0].asnumpy()
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()
        rpn_loc_target = preds[3].asnumpy()
        # rpn_loc_target_valid = rpn_loc_target[np.nonzero(rpn_loc_target)]
        # rpn_loc_target_valid_abs_mean = np.mean(np.abs(rpn_loc_target_valid))
        # rpn_loc_target_valid_mean = np.mean(rpn_loc_target_valid)
        # rpn_loc_target_valid_variance = np.var(rpn_loc_target_valid)
        dets = preds[4].asnumpy()
        dets_valid_index = np.where(dets[:, :, 0] >= 0)
        dets_valid = dets[dets_valid_index[0], dets_valid_index[1]]

        # ssd as rpn loss count
        valid_count = np.sum(cls_label >= 0)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        # in case you have a 'other' class
        label[np.where(label >= cls_prob.shape[1])] = 0
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]
        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count
        # smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += valid_count

        # rpn proposals
        rpn_rois = preds[5].asnumpy()
        rpn_score = preds[6].asnumpy()
        rpn_cid = preds[7].asnumpy()

        # rcnn_FGA_cls_loss = preds[4].asnumpy()
        # rcnn_FGA_bb8_reg_loss = preds[5].asnumpy()

        # rcnn FGA loss count
        rcnn_FGA_cls_target = preds[8].asnumpy()
        rcnn_FGA_reg_target = preds[9].asnumpy()
        rcnn_FGA_cls_prob = preds[10].asnumpy()
        # rcnn_FGA_reg_target_abs_mean = np.mean(np.abs(rcnn_FGA_reg_target))
        # rcnn_FGA_reg_target_mean = np.mean(rcnn_FGA_reg_target)
        # rcnn_FGA_reg_target_variance = np.var(rcnn_FGA_reg_target)

        # softmax version loss update
        # rcnn_valid_count = np.sum(rcnn_FGA_cls_target >= 0)
        # rcnn_FGA_cls_target = rcnn_FGA_cls_target.flatten()
        # rcnn_mask = np.where(rcnn_FGA_cls_target >= 0)[0]
        # rcnn_indices = np.int64(rcnn_FGA_cls_target[rcnn_mask])
        # rcnn_prob = rcnn_FGA_cls_prob.reshape((-1, rcnn_FGA_cls_prob.shape[2]))
        # rcnn_prob = rcnn_prob[rcnn_mask, rcnn_indices]
        # self.sum_metric[2] += (-np.log(rcnn_prob +self.eps)).sum()
        # self.num_inst[2] += rcnn_valid_count
        #
        # self.sum_metric[3] += np.sum(rcnn_FGA_bb8_reg_loss)
        # self.num_inst[3] += rcnn_valid_count

        # heatmap version loss update
        # logistic_loss = - rcnn_FGA_cls_target * np.log(rcnn_FGA_cls_prob) - (1 - rcnn_FGA_cls_target) * np.log(1 - rcnn_FGA_cls_prob)

        self.sum_metric[2] += np.mean(rcnn_FGA_cls_loss)
        self.num_inst[2] += 1

        # rcnn_FGA_bb8_reg_loss = np.sum(rcnn_FGA_bb8_reg_loss, axis=(2,3))
        self.sum_metric[3] += np.mean(rcnn_FGA_bb8_reg_loss)
        self.num_inst[3] += 1

        rcnn_FGA_cls_score = preds[12].asnumpy()
        rcnn_FGA_bb8_pred = preds[13].asnumpy()

        rpn_rois_width = rpn_rois[:, 3] - rpn_rois[:, 1]
        rpn_rois_height = rpn_rois[:, 4] - rpn_rois[:, 2]
        rcnn_FGA_reg_pred_error = np.abs((rcnn_FGA_bb8_pred - rcnn_FGA_reg_target) * 0.1).reshape(rpn_rois.shape[0], 8,
                                                                                                  2)
        rcnn_FGA_reg_pred_error_x = rcnn_FGA_reg_pred_error[:, :, 0] * rpn_rois_width[:, np.newaxis]
        rcnn_FGA_reg_pred_error_y = rcnn_FGA_reg_pred_error[:, :, 1] * rpn_rois_height[:, np.newaxis]
        rcnn_FGA_reg_mae_pixel = np.concatenate((rcnn_FGA_reg_pred_error_x[:, :, np.newaxis],
                                                 rcnn_FGA_reg_pred_error_y[:, :, np.newaxis]), axis=2)
        self.sum_metric[4] += np.sum(rcnn_FGA_reg_mae_pixel)
        self.num_inst[4] += rcnn_FGA_reg_mae_pixel.size

        rcnn_FGA_cls_target_max_index = np.argmax(rcnn_FGA_cls_target.reshape(rcnn_FGA_cls_target.shape[0],
                                                                              rcnn_FGA_cls_target.shape[1], -1)
                                                  , axis=-1)
        rcnn_FGA_cls_score_max_index = np.argmax(rcnn_FGA_cls_score.reshape(rcnn_FGA_cls_score.shape[0],
                                                                            rcnn_FGA_cls_score.shape[1], -1),
                                                 axis=-1)
        index_correct = rcnn_FGA_cls_score_max_index == rcnn_FGA_cls_target_max_index
        self.sum_metric[5] += np.sum(index_correct)
        self.num_inst[5] += rcnn_FGA_cls_target_max_index.size

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s' % (self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                      for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)
