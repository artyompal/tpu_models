"""
    SE-ResNeXt for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

# __all__ = ['SEResNeXt', 'seresnext50_32x4d', 'seresnext101_32x4d', 'seresnext101_64x4d']

import math

import numpy as np          # FIXME: do we really need numpy for tensor multiplication??
import tensorflow as tf

# from .common import conv1x1_block, se_block, is_channels_first, flatten
# from .resnet import res_init_block
# from .resnext import resnext_bottleneck


def get_activation_layer(x,
                         activation,
                         name="activ"):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    activation : function or str
        Activation function or name of activation function.
    name : str, default 'activ'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    assert (activation is not None)
    if isinstance(activation, str):
        if activation == "relu":
            x = tf.nn.relu(x, name=name)
        elif activation == "relu6":
            x = tf.nn.relu6(x, name=name)
        else:
            raise NotImplementedError()
    else:
        x = activation(x)
    return x


def is_channels_first(data_format):
    """
    Is tested data format channels first.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    bool
        A flag.
    """
    return data_format == "channels_first"


def get_channel_axis(data_format):
    """
    Get channel axis.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    int
        Channel axis.
    """
    return 1 if is_channels_first(data_format) else -1


def flatten(x,
            data_format):
    """
    Flattens the input to two dimensional.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    if not is_channels_first(data_format):
        x = tf.transpose(x, perm=(0, 3, 1, 2))
    x = tf.reshape(x, shape=(-1, np.prod(x.get_shape().as_list()[1:])))
    return x


def batchnorm(x,
              momentum=0.9,
              epsilon=1e-5,
              training=False,
              data_format="channels_last",
              name=None):
    """
    Batch normalization layer.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default None
        Layer name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = tf.keras.layers.BatchNormalization(
        axis=get_channel_axis(data_format),
        momentum=momentum,
        epsilon=epsilon,
        name=name)(
        inputs=x,
        training=training)
    return x


def maxpool2d(x,
              pool_size,
              strides,
              padding=0,
              ceil_mode=False,
              data_format="channels_last",
              name=None):
    """
    Max pooling operation for two dimensional (spatial) data.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    pool_size : int or tuple/list of 2 int
        Size of the max pooling windows.
    strides : int or tuple/list of 2 int
        Strides of the pooling.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default None
        Layer name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(padding, int):
        padding = (padding, padding)

    if ceil_mode:
        height = int(x.shape[2])
        out_height = float(height + 2 * padding[0] - pool_size[0]) / strides[0] + 1.0
        if math.ceil(out_height) > math.floor(out_height):
            padding = (padding[0] + 1, padding[1])
        width = int(x.shape[3])
        out_width = float(width + 2 * padding[1] - pool_size[1]) / strides[1] + 1.0
        if math.ceil(out_width) > math.floor(out_width):
            padding = (padding[0], padding[1] + 1)

    if (padding[0] > 0) or (padding[1] > 0):
        if is_channels_first(data_format):
            x = tf.pad(x, [[0, 0], [0, 0], list(padding), list(padding)], mode="REFLECT")
        else:
            x = tf.pad(x, [[0, 0], list(padding), list(padding), [0, 0]], mode="REFLECT")

    x = tf.keras.layers.MaxPooling2D(
        pool_size=pool_size,
        strides=strides,
        padding="valid",
        data_format=data_format,
        name=name)(x)
    return x


def avgpool2d(x,
              pool_size,
              strides,
              padding=0,
              ceil_mode=False,
              data_format="channels_last",
              name=None):
    """
    Average pooling operation for two dimensional (spatial) data.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    pool_size : int or tuple/list of 2 int
        Size of the max pooling windows.
    strides : int or tuple/list of 2 int
        Strides of the pooling.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default None
        Layer name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(padding, int):
        padding = (padding, padding)

    if ceil_mode:
        height = int(x.shape[2])
        out_height = float(height + 2 * padding[0] - pool_size[0]) / strides[0] + 1.0
        if math.ceil(out_height) > math.floor(out_height):
            padding = (padding[0] + 1, padding[1])
        width = int(x.shape[3])
        out_width = float(width + 2 * padding[1] - pool_size[1]) / strides[1] + 1.0
        if math.ceil(out_width) > math.floor(out_width):
            padding = (padding[0], padding[1] + 1)

    if (padding[0] > 0) or (padding[1] > 0):
        if is_channels_first(data_format):
            x = tf.pad(x, [[0, 0], [0, 0], list(padding), list(padding)], mode="CONSTANT")
        else:
            x = tf.pad(x, [[0, 0], list(padding), list(padding), [0, 0]], mode="CONSTANT")

    x = tf.keras.layers.AveragePooling2D(
        pool_size=pool_size,
        strides=1,
        padding="valid",
        data_format=data_format,
        name=name)(x)

    if (strides[0] > 1) or (strides[1] > 1):
        x = tf.keras.layers.AveragePooling2D(
            pool_size=1,
            strides=strides,
            padding="valid",
            data_format=data_format,
            name=name + "/stride")(x)
    return x


def conv2d(x,
           in_channels,
           out_channels,
           kernel_size,
           strides=1,
           padding=0,
           dilation=1,
           groups=1,
           use_bias=True,
           data_format="channels_last",
           name="conv2d"):
    """
    Convolution 2D layer wrapper.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'conv2d'
        Layer name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if (padding[0] > 0) or (padding[1] > 0):
        if is_channels_first(data_format):
            paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
        else:
            paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]
        x = tf.pad(x, paddings=paddings_tf)

    if groups == 1:
        x = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            data_format=data_format,
            dilation_rate=dilation,
            use_bias=use_bias,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
            name=name)(x)
    elif (groups == out_channels) and (out_channels == in_channels):
        assert (dilation[0] == 1) and (dilation[1] == 1)
        kernel = tf.get_variable(
            name=name + "/dw_kernel",
            shape=kernel_size + (in_channels, 1),
            initializer=tf.variance_scaling_initializer(2.0))
        x = tf.nn.depthwise_conv2d(
            input=x,
            filter=kernel,
            strides=(1, 1) + strides if is_channels_first(data_format) else (1,) + strides + (1,),
            padding="VALID",
            rate=(1, 1),
            name=name,
            data_format="NCHW" if is_channels_first(data_format) else "NHWC")
        if use_bias:
            raise NotImplementedError
    else:
        assert (in_channels % groups == 0)
        assert (out_channels % groups == 0)
        in_group_channels = in_channels // groups
        out_group_channels = out_channels // groups
        group_list = []
        for gi in range(groups):
            if is_channels_first(data_format):
                xi = x[:, gi * in_group_channels:(gi + 1) * in_group_channels, :, :]
            else:
                xi = x[:, :, :, gi * in_group_channels:(gi + 1) * in_group_channels]
            xi = tf.keras.layers.Conv2D(
                filters=out_group_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                data_format=data_format,
                dilation_rate=dilation,
                use_bias=use_bias,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
                name=name + "/convgroup{}".format(gi + 1))(xi)
            group_list.append(xi)
        x = tf.concat(group_list, axis=get_channel_axis(data_format), name=name + "/concat")

    return x


def conv1x1(x,
            in_channels,
            out_channels,
            strides=1,
            groups=1,
            use_bias=False,
            data_format="channels_last",
            name="conv1x1"):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'conv1x1'
        Layer name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        groups=groups,
        use_bias=use_bias,
        data_format=data_format,
        name=name)


def conv3x3(x,
            in_channels,
            out_channels,
            strides=1,
            padding=1,
            groups=1,
            use_bias=False,
            data_format="channels_last",
            name="conv3x3"):
    """
    Convolution 3x3 layer.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'conv3x3'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        groups=groups,
        use_bias=use_bias,
        data_format=data_format,
        name=name)


def conv_block(x,
               in_channels,
               out_channels,
               kernel_size,
               strides,
               padding,
               dilation=1,
               groups=1,
               use_bias=False,
               use_bn=True,
               activation="relu",
               training=False,
               data_format="channels_last",
               name="conv_block"):
    """
    Standard convolution block with Batch normalization and activation.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'conv_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        data_format=data_format,
        name=name + "/conv")
    if use_bn:
        x = batchnorm(
            x=x,
            training=training,
            data_format=data_format,
            name=name + "/bn")
    if activation is not None:
        x = get_activation_layer(
            x=x,
            activation=activation,
            name=name + "/activ")
    return x


def conv1x1_block(x,
                  in_channels,
                  out_channels,
                  strides=1,
                  groups=1,
                  use_bias=False,
                  activation="relu",
                  training=False,
                  data_format="channels_last",
                  name="conv1x1_block"):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'conv1x1_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=groups,
        use_bias=use_bias,
        activation=activation,
        training=training,
        data_format=data_format,
        name=name)


def conv3x3_block(x,
                  in_channels,
                  out_channels,
                  strides=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  activation="relu",
                  training=False,
                  data_format="channels_last",
                  name="conv3x3_block"):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'conv3x3_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        activation=activation,
        training=training,
        data_format=data_format,
        name=name)


def conv5x5_block(x,
                  in_channels,
                  out_channels,
                  strides=1,
                  padding=2,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  activation="relu",
                  training=False,
                  data_format="channels_last",
                  name="conv3x3_block"):
    """
    5x5 version of the standard convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 2
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'conv3x3_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        activation=activation,
        training=training,
        data_format=data_format,
        name=name)


def conv7x7_block(x,
                  in_channels,
                  out_channels,
                  strides=1,
                  padding=3,
                  use_bias=False,
                  activation="relu",
                  training=False,
                  data_format="channels_last",
                  name="conv7x7_block"):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'conv7x7_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        activation=activation,
        training=training,
        data_format=data_format,
        name=name)

def res_init_block(x,
                   in_channels,
                   out_channels,
                   training,
                   data_format,
                   name):
    """
    ResNet specific initial block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'res_init_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv7x7_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=2,
        training=training,
        data_format=data_format,
        name=name + "/conv")
    x = maxpool2d(
        x=x,
        pool_size=3,
        strides=2,
        padding=1,
        data_format=data_format,
        name=name + "/pool")
    return x


def resnext_bottleneck(x,
                       in_channels,
                       out_channels,
                       strides,
                       cardinality,
                       bottleneck_width,
                       bottleneck_factor=4,
                       training=False,
                       data_format="channels_last",
                       name="resnext_bottleneck"):
    """
    ResNeXt bottleneck block for residual path in ResNeXt unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'resnext_bottleneck'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    mid_channels = out_channels // bottleneck_factor
    D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
    group_width = cardinality * D

    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=group_width,
        training=training,
        data_format=data_format,
        name=name + "/conv1")
    x = conv3x3_block(
        x=x,
        in_channels=group_width,
        out_channels=group_width,
        strides=strides,
        groups=cardinality,
        training=training,
        data_format=data_format,
        name=name + "/conv2")
    x = conv1x1_block(
        x=x,
        in_channels=group_width,
        out_channels=out_channels,
        activation=None,
        training=training,
        data_format=data_format,
        name=name + "/conv3")
    return x


def se_block(x,
             channels,
             reduction=16,
             activation="relu",
             data_format="channels_last",
             name="se_block"):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    activation : function or str, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'se_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    assert(len(x.shape) == 4)
    mid_cannels = channels // reduction
    pool_size = x.shape[2:4] if is_channels_first(data_format) else x.shape[1:3]

    w = tf.keras.layers.AveragePooling2D(
        pool_size=pool_size,
        strides=1,
        data_format=data_format,
        name=name + "/pool")(x)
    w = conv1x1(
        x=w,
        in_channels=channels,
        out_channels=mid_cannels,
        use_bias=True,
        data_format=data_format,
        name=name + "/conv1/conv")
    w = get_activation_layer(
        x=w,
        activation=activation,
        name=name + "/activ")
    w = conv1x1(
        x=w,
        in_channels=mid_cannels,
        out_channels=channels,
        use_bias=True,
        data_format=data_format,
        name=name + "/conv2/conv")
    w = tf.nn.sigmoid(w, name=name + "/sigmoid")
    x = x * w
    return x



def seresnext_unit(x,
                   in_channels,
                   out_channels,
                   strides,
                   cardinality,
                   bottleneck_width,
                   training,
                   data_format,
                   name="seresnext_unit"):
    """
    SE-ResNeXt unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'seresnext_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    resize_identity = (in_channels != out_channels) or (strides != 1)
    if resize_identity:
        identity = conv1x1_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            activation=None,
            training=training,
            data_format=data_format,
            name=name + "/identity_conv")
    else:
        identity = x

    x = resnext_bottleneck(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
        training=training,
        data_format=data_format,
        name=name + "/body")

    x = se_block(
        x=x,
        channels=out_channels,
        data_format=data_format,
        name=name + "/se")

    x = x + identity

    x = tf.nn.relu(x, name=name + "/activ")
    return x


class SEResNeXt(object):
    """
    SE-ResNeXt model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    in_channels : int, default 3
        Number of input channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 blocks,
                 cardinality=32,
                 bottleneck_width=4,
                 in_channels=3,
                 data_format="channels_last",
                 **kwargs):
        super(SEResNeXt, self).__init__()

        if blocks == 50:
            layers = [3, 4, 6, 3]
        elif blocks == 101:
            layers = [3, 4, 23, 3]
        else:
            raise ValueError("Unsupported SE-ResNeXt with number of blocks: {}".format(blocks))

        init_block_channels = 64
        channels_per_layers = [256, 512, 1024, 2048]

        channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_channels = in_channels
        self.data_format = data_format

        self._gen_fn = self._generator_fn()

    def __call__(self,
                 x,
                 is_training=False):
        return self._gen_fn(x, is_training)

    def _generator_fn(self):
        def model(x, is_training=False):
            """
            Build a model graph.

            Parameters:
            ----------
            x : Tensor
                Input tensor.
            is_training : bool, or a TensorFlow boolean scalar tensor, default False
              Whether to return the output in training mode or in inference mode.

            Returns
            -------
            Dictionary: int to tensor
                Dictionary of features at various scales
            """
            in_channels = self.in_channels
            x = res_init_block(
                x=x,
                in_channels=in_channels,
                out_channels=self.init_block_channels,
                training=is_training,
                data_format=self.data_format,
                name="features/init_block")

            in_channels = self.init_block_channels
            res = {}

            for i, channels_per_stage in enumerate(self.channels):
                for j, out_channels in enumerate(channels_per_stage):
                    strides = 2 if (j == 0) and (i != 0) else 1
                    x = seresnext_unit(
                        x=x,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        strides=strides,
                        cardinality=self.cardinality,
                        bottleneck_width=self.bottleneck_width,
                        training=is_training,
                        data_format=self.data_format,
                        name="features/stage{}/unit{}".format(i + 1, j + 1))
                    in_channels = out_channels

                res[i + 2] = x

            return res

        return model

        # x = tf.keras.layers.AveragePooling2D(
        #     pool_size=7,
        #     strides=1,
        #     data_format=self.data_format,
        #     name="features/final_pool")(x)
        #
        # # x = tf.layers.flatten(x)
        # x = flatten(
        #     x=x,
        #     data_format=self.data_format)
        # x = tf.keras.layers.Dense(
        #     units=self.classes,
        #     name="output")(x)
        #
        # return x


# def get_seresnext(blocks,
#                   cardinality,
#                   bottleneck_width,
#                   model_name=None,
#                   **kwargs):
#     """
#     Create SE-ResNeXt model with specific parameters.
#
#     Parameters:
#     ----------
#     blocks : int
#         Number of blocks.
#     cardinality: int
#         Number of groups.
#     bottleneck_width: int
#         Width of bottleneck block.
#     model_name : str or None, default None
#         Model name for loading pretrained model.
#
#     Returns
#     -------
#     functor
#         Functor for model graph creation with extra fields.
#     """
#
#     if blocks == 50:
#         layers = [3, 4, 6, 3]
#     elif blocks == 101:
#         layers = [3, 4, 23, 3]
#     else:
#         raise ValueError("Unsupported SE-ResNeXt with number of blocks: {}".format(blocks))
#
#     init_block_channels = 64
#     channels_per_layers = [256, 512, 1024, 2048]
#
#     channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
#
#     net = SEResNeXt(
#         channels=channels,
#         init_block_channels=init_block_channels,
#         cardinality=cardinality,
#         bottleneck_width=bottleneck_width,
#         **kwargs)
#
#     # if pretrained:
#     #     if (model_name is None) or (not model_name):
#     #         raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
#     #     from .model_store import download_state_dict
#     #     net.state_dict, net.file_path = download_state_dict(
#     #         model_name=model_name,
#     #         local_model_store_dir_path=root)
#     # else:
#     #     net.state_dict = None
#     #     net.file_path = None
#
#     return net
#
#
# def seresnext50_32x4d(**kwargs):
#     """
#     SE-ResNeXt-50 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
#
#     Parameters:
#     ----------
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     root : str, default '~/.tensorflow/models'
#         Location for keeping the model parameters.
#
#     Returns
#     -------
#     functor
#         Functor for model graph creation with extra fields.
#     """
#     return get_seresnext(blocks=50, cardinality=32, bottleneck_width=4, model_name="seresnext50_32x4d", **kwargs)
#
#
# def seresnext101_32x4d(**kwargs):
#     """
#     SE-ResNeXt-101 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
#
#     Parameters:
#     ----------
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     root : str, default '~/.tensorflow/models'
#         Location for keeping the model parameters.
#
#     Returns
#     -------
#     functor
#         Functor for model graph creation with extra fields.
#     """
#     return get_seresnext(blocks=101, cardinality=32, bottleneck_width=4, model_name="seresnext101_32x4d", **kwargs)
#
#
# def seresnext101_64x4d(**kwargs):
#     """
#     SE-ResNeXt-101 (64x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
#
#     Parameters:
#     ----------
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     root : str, default '~/.tensorflow/models'
#         Location for keeping the model parameters.
#
#     Returns
#     -------
#     functor
#         Functor for model graph creation with extra fields.
#     """
#     return get_seresnext(blocks=101, cardinality=64, bottleneck_width=4, model_name="seresnext101_64x4d", **kwargs)
#
