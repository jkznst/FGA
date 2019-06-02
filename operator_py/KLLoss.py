# import numpy as np
import mxnet as mx

class KLLossOperator(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, normalization, grad_scale):
        super(KLLossOperator, self).__init__()
        self._normalization = normalization
        self._grad_scale = float(grad_scale)
        self.eps = 1e-14

        # print(self._ignore_label)
        # print(self._normalization == "valid")
        # print(self._grad_scale)

    def forward(self, is_train, req, in_data, out_data, aux):
        reg_pred = in_data[0]
        reg_target = in_data[1]
        reg_weight = in_data[2]
        variance = in_data[3]

        abs_error = mx.nd.abs(reg_pred - reg_target)
        condition = abs_error > 1.
        greater_loss = mx.nd.exp(-variance) * (abs_error - 0.5) + 0.5 * variance
        lower_loss = 0.5 * mx.nd.exp(-variance) * mx.nd.square(abs_error) + 0.5 * variance
        kl_loss = mx.nd.where(condition, greater_loss, lower_loss)
        kl_loss = kl_loss * reg_weight

        self.assign(out_data[0], req[0], kl_loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        reg_pred = in_data[0]
        reg_target = in_data[1]
        reg_weight = in_data[2]
        variance = in_data[3]

        condition = mx.nd.abs(reg_pred - reg_target) > 1.
        greater_d_reg_pred = mx.nd.exp(-variance) * mx.nd.sign(reg_pred - reg_target)
        lower_d_reg_pred = mx.nd.exp(-variance) * (reg_pred - reg_target)
        d_reg_pred = mx.nd.where(condition, greater_d_reg_pred, lower_d_reg_pred)

        greater_d_variance = -(mx.nd.abs(reg_pred - reg_target) - 0.5) * mx.nd.exp(-variance) + 0.5
        lower_d_variance = -0.5 * mx.nd.square(reg_pred - reg_target) * mx.nd.exp(-variance) + 0.5
        d_variance = mx.nd.where(condition, greater_d_variance, lower_d_variance)

        d_reg_pred = reg_weight * d_reg_pred
        d_variance = reg_weight * d_variance
        # hard sample mining
        # max_target = mx.nd.max_axis(data=cls_target, axis=-1, keepdims=True)
        # valid_c = mx.nd.sum(max_target > 0)
        # reweight = mx.nd.where(max_target > 0, 1 - max_target, mx.nd.zeros_like(max_target))
        # reweight = valid_c / mx.nd.sum(reweight)
        # d_cls_score = reweight * d_cls_score * (1 - max_target)

        # filter out invalid label
        valid_condition = (reg_weight > 0.)
        # print(valid_condition[0:100])
        # print(self._grad_scale)
        # d_cls_score = mx.nd.where(valid_condition, d_cls_score, mx.nd.zeros_like(d_cls_score))

        if self._normalization == "valid":
            # normalize the grad based on the number of valid instances
            valid_count = mx.nd.sum(valid_condition).asscalar()
            # print("valid_count:", valid_count, "  max_d_cls_score:", mx.nd.max(d_cls_score).asscalar(),
            #       "  min_d_cls_score:", mx.nd.min(d_cls_score).asscalar())
            d_reg_pred /= max(1.0, valid_count)
            d_variance /= max(1.0, valid_count)
        elif self._normalization == "batch":
            d_reg_pred /= reg_target.shape[0]
            d_variance /= reg_target.shape[0]
        elif self._normalization == "null":
            pass
        else:
            print("Unsupported normalization!!")

        if self._grad_scale is not None:
            d_reg_pred *= self._grad_scale
            d_variance *= self._grad_scale

        # if (mx.nd.max(cls_target) <= 1.0 and mx.nd.min(cls_target) >= -1.0):
        #     self.assign(in_grad[0], req[0], d_cls_score)
        # else:
        #     print("Due to custom operator error, cls_target out of range!!")
        #     print(cls_target[0:10])
        #     self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[0], req[0], d_reg_pred)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], d_variance)

@mx.operator.register("klloss")
class KLLossProp(mx.operator.CustomOpProp):
    def __init__(self, normalization, grad_scale):
        super(KLLossProp, self).__init__(need_top_grad=False) #################
        self._normalization = str(normalization)
        self._grad_scale = float(grad_scale)

        # print(self._ignore_label)
        # print(self._normalization)
        # print(self._grad_scale)

    def list_arguments(self):
        return ['reg_pred', 'reg_target', 'reg_weight', 'variance']

    def list_outputs(self):
        return ['kl_loss']

    def infer_shape(self, in_shape):
        reg_pred_shape = in_shape[0]
        reg_target_shape = in_shape[1]
        reg_weight_shape = in_shape[2]
        variance_shape = in_shape[3]

        kl_loss_shape = reg_pred_shape

        return [reg_pred_shape, reg_target_shape, reg_weight_shape, variance_shape], [kl_loss_shape, ], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return KLLossOperator(self._normalization, self._grad_scale)

    # def declare_backward_dependency(self, out_grad, in_data, out_data):
    #     return []