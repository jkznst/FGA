import numpy as np
import mxnet as mx

class WeightedCELossOperator(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, ignore_label, normalization, grad_scale):
        super(WeightedCELossOperator, self).__init__()
        self._ignore_label = np.int(ignore_label)
        self._normalization = normalization
        self._grad_scale = np.float(grad_scale)
        self.eps = 1e-14

        # print(self._ignore_label)
        # print(self._normalization == "valid")
        # print(self._grad_scale)

    def forward(self, is_train, req, in_data, out_data, aux):
        cls_score = in_data[0]
        y = mx.nd.softmax(cls_score, axis=-1)
        self._prob = y
        self._cls_target = in_data[1].astype("int32")
        self._reg_smoothl1 = in_data[2]

        # print(self._prob[0:10])
        # print(self._cls_target[0:10])
        # print(self._reg_smoothl1[0:10])

        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        cls_prob = self._prob    # shape (N, 8, 4)
        cls_target = self._cls_target        # shape (N, 8)
        reg_smooth_l1 = self._reg_smoothl1      # shape (N, 64)

        # print(cls_prob[0:10])
        # print(cls_target[0:10])
        # print(reg_smooth_l1[0:10])

        one_hot_label = mx.nd.one_hot(cls_target, depth=cls_prob.shape[2], on_value=1, off_value=0)
        d_cls_score = cls_prob - one_hot_label

        # reweight cls grad by reg smooth l1
        reg_smooth_l1 = mx.nd.reshape(reg_smooth_l1, shape=(cls_prob.shape[0], cls_prob.shape[1], -1))
        reg_smooth_l1 = mx.nd.max(mx.nd.abs(reg_smooth_l1), axis=-1)
        # print(reg_smooth_l1[0:10])
        weight = mx.nd.where(cls_target != self._ignore_label, 2. / (mx.nd.exp(reg_smooth_l1) + 1.), mx.nd.zeros_like(reg_smooth_l1))
        # print(weight[0:10])
        valid_count = mx.nd.sum(weight != 0)
        weight_valid_mean = mx.nd.sum(weight) / valid_count
        # print("weight_valid_mean:", weight_valid_mean)
        # print("reg_smoothl1_valid_mean:", mx.nd.sum(reg_smooth_l1) / valid_count)
        weight /= weight_valid_mean
        # print(w1feight)
        d_cls_score *= mx.nd.expand_dims(weight, axis=-1)

        # filter out invalid label
        valid_condition = (cls_target != self._ignore_label)
        # print(valid_condition[0:10])
        # print(self._ignore_label)
        d_cls_score *= mx.nd.expand_dims(valid_condition.astype('float32'), axis=-1)

        if self._normalization == "valid":
            # normalize the grad based on the number of valid instances
            valid_count = mx.nd.sum(valid_condition).asscalar()
            # print("valid_count:", valid_count)
            d_cls_score /= max(1.0, valid_count)
        elif self._normalization == "batch":
            d_cls_score /= cls_target.shape[0]
        elif self._normalization == "null":
            pass
        else:
            print("Unsurpported normalization!!")

        if self._grad_scale is not None:
            d_cls_score *= self._grad_scale

        self.assign(in_grad[0], req[0], d_cls_score)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

@mx.operator.register("weightedcrossentropyloss")
class WeightedCELossProp(mx.operator.CustomOpProp):
    def __init__(self, ignore_label, normalization, grad_scale):
        super(WeightedCELossProp, self).__init__(need_top_grad=False) #################
        self._ignore_label = int(ignore_label)
        self._normalization = str(normalization)
        self._grad_scale = float(grad_scale)

        # print(self._ignore_label)
        # print(self._normalization)
        # print(self._grad_scale)

    def list_arguments(self):
        return ['cls_score', 'label', 'reg_smoothl1']

    def list_outputs(self):
        return ['cls_prob']

    def infer_shape(self, in_shape):
        cls_score_shape = in_shape[0]
        label_shape = in_shape[1]
        reg_smoothl1_shape = in_shape[2]

        cls_prob_shape = cls_score_shape

        return [cls_score_shape, label_shape, reg_smoothl1_shape], [cls_prob_shape, ], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return WeightedCELossOperator(self._ignore_label, self._normalization, self._grad_scale)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []