"""Presets for various network configurations"""
import logging
from symbol import symbol_builder
import numpy as np

def get_scales(min_scale=0.2, max_scale=0.9,num_layers=6):
    """ Following the ssd arxiv paper, regarding the calculation of scales & ratios

    Parameters
    ----------
    min_scale : float
    max_scales: float
    num_layers: int
        number of layers that will have a detection head
    anchor_ratios: list
    first_layer_ratios: list

    return
    ------
    sizes : list
        list of scale sizes per feature layer
    ratios : list
        list of anchor_ratios per feature layer
    """

    # this code follows the original implementation of wei liu
    # for more, look at ssd/score_ssd_pascal.py:310 in the original caffe implementation
    min_ratio = int(min_scale * 100)
    max_ratio = int(max_scale * 100)
    step = int(np.floor((max_ratio - min_ratio) / (num_layers - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(ratio / 100.)
        max_sizes.append((ratio + step) / 100.)
    min_sizes = [int(100*min_scale / 2.0) / 100.0] + min_sizes
    max_sizes = [min_scale] + max_sizes

    # convert it back to this implementation's notation:
    scales = []
    for layer_idx in range(num_layers):
        scales.append([min_sizes[layer_idx], np.single(np.sqrt(min_sizes[layer_idx] * max_sizes[layer_idx]))])
    return scales

def get_config(network, data_shape, **kwargs):
    """Configuration factory for various networks

    Parameters
    ----------
    network : str
        base network name, such as vgg_reduced, inceptionv3, resnet...
    data_shape : int
        input data dimension
    kwargs : dict
        extra arguments
    """
    if network == 'vgg16_reduced':
        if data_shape >= 448:
            from_layers = ['relu4_3', 'relu7', '', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256, 256]
            strides = [-1, -1, 2, 2, 2, 2, 1]
            pads = [-1, -1, 1, 1, 1, 1, 1]
            sizes = get_scales(min_scale=0.15, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 512 else [x / 512.0 for x in
                [8, 16, 32, 64, 128, 256, 512]]
        else:
            from_layers = ['relu4_3', 'relu7', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256]
            strides = [-1, -1, 2, 2, 1, 1]
            pads = [-1, -1, 1, 1, 0, 0]
            sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 300 else [x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
        if not (data_shape == 300 or data_shape == 512):
            logging.warn('data_shape %d was not tested, use with caucious.' % data_shape)
        return locals()
    elif network == 'resnet50':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnet'
        from_layers = ['P3', 'P4', 'P5', 'P6', 'P7']
        num_filters = [-1, -1, -1, -1, -1]
        strides = [-1, -1, -1, -1, -1]
        pads = [-1, -1, -1, -1, -1]
        sizes = get_scales(min_scale=0.1, max_scale=0.5, num_layers=len(from_layers))
        ratios = [[1, 2, .5], [1, 2, .5], [1, 2, .5], [1, 2, .5], \
                  [1, 2, .5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50d':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnet'
        from_layers = ['P3', 'P4', 'P5', 'P6', 'P7']
        num_filters = [-1, -1, -1, -1, -1]
        strides = [-1, -1, -1, -1, -1]
        pads = [-1, -1, -1, -1, -1]
        sizes = get_scales(min_scale=0.1, max_scale=0.5, num_layers=len(from_layers))
        ratios = [[1, 2, .5], [1, 2, .5], [1, 2, .5], [1, 2, .5], \
                  [1, 2, .5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50deeplabv2':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnet'
        from_layers = ['P3', 'P4', 'P5', 'P6', 'P7']
        num_filters = [-1, -1, -1, -1, -1]
        strides = [-1, -1, -1, -1, -1]
        pads = [-1, -1, -1, -1, -1]
        sizes = get_scales(min_scale=0.1, max_scale=0.5, num_layers=len(from_layers))
        ratios = [[1, 2, .5], [1, 2, .5], [1, 2, .5], [1, 2, .5], \
                  [1, 2, .5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50m':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnetm'
        from_layers = ['P3', 'P4', 'P5', 'P6', 'P7']
        num_filters = [-1, -1, -1, -1, -1]
        strides = [-1, -1, -1, -1, -1]
        pads = [-1, -1, -1, -1, -1]
        sizes = get_scales(min_scale=0.1, max_scale=0.5, num_layers=len(from_layers))
        ratios = [[1, 2, .5], [1, 2, .5], [1, 2, .5], [1, 2, .5], \
                  [1, 2, .5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50md':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnetm'
        from_layers = ['P3', 'P4', 'P5', 'P6', 'P7']
        num_filters = [-1, -1, -1, -1, -1]
        strides = [-1, -1, -1, -1, -1]
        pads = [-1, -1, -1, -1, -1]
        sizes = get_scales(min_scale=0.1, max_scale=0.5, num_layers=len(from_layers))
        ratios = [[1, 2, .5], [1, 2, .5], [1, 2, .5], [1, 2, .5], \
                  [1, 2, .5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['P2', 'P3', 'P4', 'P5', 'P6']
        num_filters = [-1, -1, -1, -1, -1]
        strides = [-1, -1, -1, -1, -1]
        pads = [-1, -1, -1, -1, -1]
        sizes = get_scales(min_scale=0.2, max_scale=0.8, num_layers=len(from_layers))
        ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], \
                  [1, 2, .5]]
        normalizations = -1
        steps = []
        return locals()
    # elif network == 'resnet50':
    #     num_layers = 50
    #     image_shape = '3,224,224'  # resnet require it as shape check
    #     network = 'resnet'
    #     from_layers = ['_plus12', '_plus15', '', '', '', '']
    #     num_filters = [-1, -1, 512, 256, 256, 128]
    #     strides = [-1, -1, 2, 2, 2, 2]
    #     pads = [-1, -1, 1, 1, 1, 1]
    #     sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
    #     ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
    #         [1,2,.5], [1,2,.5]]
    #     normalizations = -1
    #     steps = []
    #     return locals()
    # elif network == 'resnet50e':
    #     num_layers = 50
    #     image_shape = '3,224,224'  # resnet require it as shape check
    #     network = 'resnet'
    #     from_layers = ['_plus6', '_plus12', '_plus15', '', '', '']
    #     num_filters = [-1, -1, -1, 512, 256, 256]
    #     strides = [-1, -1, -1, 2, 2, 2]
    #     pads = [-1, -1, -1, 1, 1, 1]
    #     sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
    #     ratios = [[1,2,.5], [1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
    #         [1,2,.5]]
    #     normalizations = -1
    #     steps = []
    #     return locals()
    # elif network == 'resnet50m':
    #     num_layers = 50
    #     image_shape = '3,224,224'  # resnet require it as shape check
    #     network = 'resnetm'
    #     from_layers = ['_plus12', '_plus15', '', '', '', '']
    #     num_filters = [-1, -1, 512, 256, 256, 128]
    #     strides = [-1, -1, 2, 2, 2, 2]
    #     pads = [-1, -1, 1, 1, 1, 1]
    #     sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
    #     ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
    #         [1,2,.5], [1,2,.5]]
    #     normalizations = -1
    #     steps = []
    #     return locals()
    # elif network == 'resnet101':
    #     num_layers = 101
    #     image_shape = '3,224,224'
    #     network = 'resnet'
    #     from_layers = ['_plus29', '_plus32', '', '', '', '']
    #     num_filters = [-1, -1, 512, 256, 256, 128]
    #     strides = [-1, -1, 2, 2, 2, 2]
    #     pads = [-1, -1, 1, 1, 1, 1]
    #     sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
    #     ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
    #         [1,2,.5], [1,2,.5]]
    #     normalizations = -1
    #     steps = []
    #     return locals()
    # elif network == 'mobilenet':
    #     from_layers = ['conv_12_relu', 'conv_14_relu', '', '', '', '', '']
    #     num_filters = [-1, -1, 512, 256, 256, 256, 256]
    #     strides = [-1, -1, 2, 2, 2, 2, 2]
    #     pads = [-1, -1, 1, 1, 1, 1, 1]
    #     sizes = get_scales(min_scale=0.15, max_scale=0.9, num_layers=len(from_layers))
    #     ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
    #               [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
    #     normalizations = -1
    #     steps = []
    #     return locals()
    # elif network == 'densenet121':
    #     network = 'densenet'
    #     data_type = 'imagenet'
    #     units = [6, 12, 24, 16]
    #     num_stage = 4
    #     growth_rate = 32
    #     bottle_neck = True
    #     from_layers = ['DBstage3_concat24', 'DBstage4_concat16', '', '', '', '']
    #     num_filters = [-1, -1, 256, 256, 256, 128]
    #     strides = [-1, -1, 2, 2, 2, 2]
    #     pads = [-1, -1, 1, 1, 1, 1]
    #     sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
    #     ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
    #         [1,2,.5], [1,2,.5]]
    #     normalizations = -1
    #     steps = []
    #     return locals()
    # elif network == 'densenet-tiny':
    #     network = 'densenet'
    #     data_type = 'imagenet'
    #     units = [6, 12, 18, 12]
    #     num_stage = 4
    #     growth_rate = 16
    #     bottle_neck = True
    #     from_layers = ['DBstage2_concat12', 'DBstage3_concat18', '', '', '', '']
    #     num_filters = [-1, -1, 256, 256, 256, 128]
    #     strides = [-1, -1, 2, 2, 2, 2]
    #     pads = [-1, -1, 1, 1, 1, 1]
    #     sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
    #     ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
    #         [1,2,.5], [1,2,.5]]
    #     normalizations = -1
    #     steps = []
    #     return locals()
    else:
        msg = 'No configuration found for %s with data_shape %d' % (network, data_shape)
        raise NotImplementedError(msg)

def get_symbol_train(network, mode, data_shape, alpha_bb8, **kwargs):
    """Wrapper for get symbol for train

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol_train for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol_train(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    networks = {
        'vgg16': symbol_builder.get_vgg_reduced_fpn_train,
        'resnet50': getattr(symbol_builder, 'get_{}_resnet_fpn_train'.format(mode)),
        'resnet50d': symbol_builder.get_resnetd_fpn_train,
        'resnet50deeplabv2': symbol_builder.get_resnetdeeplabv2_fpn_train,
        'resnet50m':getattr(symbol_builder, 'get_{}_resnetm_fpn_train'.format(mode)),
        'resnet50md': symbol_builder.get_resnetmd_fpn_train,
        'resnet101': symbol_builder.get_resnet_fpn_train
    }
    if network not in networks:
        raise ValueError("network {} not supported".format(network))
    return networks[network](alpha_bb8=alpha_bb8, im_info=(data_shape, data_shape, 1), **config)

    # return symbol_builder.get_symbol_train(alpha_bb8=alpha_bb8, **config)

def get_symbol(network, mode, data_shape, **kwargs):
    """Wrapper for get symbol for test

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    networks = {
        # 'vgg16': symbol_builder.get_vgg_reduced_fpn_test,
        'resnet50': getattr(symbol_builder, 'get_{}_resnet_fpn_test'.format(mode)),
        # 'resnet50d': symbol_builder.get_resnetd_fpn_test,
        # 'resnet50deeplabv2': symbol_builder.get_resnetdeeplabv2_fpn_test,
        'resnet50m': getattr(symbol_builder, 'get_{}_resnetm_fpn_test'.format(mode)),
        # 'resnet50md': symbol_builder.get_resnetmd_fpn_test,
        # 'resnet101': symbol_builder.get_resnet_fpn_test
    }
    if network not in networks:
        raise ValueError("network {} not supported".format(network))
    return networks[network](im_info=(data_shape, data_shape, 1), **config)
    # return symbol_builder.get_symbol(**config)
