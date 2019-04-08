import numpy as np
import mxnet as mx

class SoftCELossOperator(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, ignore_label, normalization, grad_scale):
        super(SoftCELossOperator, self).__init__()
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
        self._cls_target = in_data[1].astype("float32")
        # print("max_cls_target_in_loss:", self._cls_target.max().asscalar(), "min_cls_target_in_loss:",
        #       self._cls_target.min().asscalar())

        # print(self._prob[0:10])
        # print(self._cls_target[0:10])
        # print(self._reg_smoothl1[0:10])

        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        cls_prob = self._prob    # shape (N, 8, 4)
        cls_target = self._cls_target        # shape (N, 8, 4)

        # print("max_cls_target_in_loss:", cls_target.max().asscalar(), "min_cls_target_in_loss:", cls_target.min().asscalar())

        # print(cls_prob[0:10])
        # print(cls_target[0:10])

        d_cls_score = cls_prob - cls_target

        # filter out invalid label
        valid_condition = (cls_target != self._ignore_label)
        # print(valid_condition[0:100])
        # print(self._grad_scale)
        d_cls_score = mx.nd.where(valid_condition, d_cls_score, mx.nd.zeros_like(d_cls_score))

        if self._normalization == "valid":
            # normalize the grad based on the number of valid instances
            valid_count = mx.nd.sum(valid_condition / cls_target.shape[2]).asscalar()
            # print("valid_count:", valid_count, "  max_d_cls_score:", mx.nd.max(d_cls_score).asscalar(),
            #       "  min_d_cls_score:", mx.nd.min(d_cls_score).asscalar())
            d_cls_score /= max(1.0, valid_count)
        elif self._normalization == "batch":
            d_cls_score /= cls_target.shape[0]
        elif self._normalization == "null":
            pass
        else:
            print("Unsupported normalization!!")

        if self._grad_scale is not None:
            d_cls_score *= self._grad_scale

        if (mx.nd.max(cls_target) <= 1.0 and mx.nd.min(cls_target) >= -1.0):
            self.assign(in_grad[0], req[0], d_cls_score)
        else:
            print("Due to custom operator error, cls_target out of range!!")
            self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register("softcrossentropyloss")
class SoftCELossProp(mx.operator.CustomOpProp):
    def __init__(self, ignore_label, normalization, grad_scale):
        super(SoftCELossProp, self).__init__(need_top_grad=False) #################
        self._ignore_label = int(ignore_label)
        self._normalization = str(normalization)
        self._grad_scale = float(grad_scale)

        # print(self._ignore_label)
        # print(self._normalization)
        # print(self._grad_scale)

    def list_arguments(self):
        return ['cls_score', 'label']

    def list_outputs(self):
        return ['cls_prob']

    def infer_shape(self, in_shape):
        cls_score_shape = in_shape[0]
        label_shape = in_shape[1]

        cls_prob_shape = cls_score_shape

        return [cls_score_shape, label_shape], [cls_prob_shape, ], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return SoftCELossOperator(self._ignore_label, self._normalization, self._grad_scale)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []