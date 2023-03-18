# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
import numpy as np
import tensorflow as tf
from tf_ops.grouping.tf_grouping import knn_point_2
from tf_lib.gcn_lib import vertex
from tf_lib.gcn_lib.vertex import nodeshuffle, multi_cnn, duplicate, mlpshuffle, inception_densegcn, densegcn, gcn


def mlp(features,
        layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs,
             layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv2d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv2d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs


def up_block(inputs, up_ratio, scope='up_block', is_training=True, bn_decay=None):
    from Common.model_utils import gen_grid
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = inputs
        dim = inputs.get_shape()[-1]
        out_dim = dim * up_ratio
        grid = gen_grid(up_ratio)
        grid = tf.tile(tf.expand_dims(grid, 0),
                       [tf.shape(net)[0], 1, tf.shape(net)[1]])  # [batch_size, num_point*4, 2])
        grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])
        # grid = tf.expand_dims(grid, axis=2)

        net = tf.tile(net, [1, up_ratio, 1, 1])
        net = tf.concat([net, grid], axis=-1)

        net = attention_unit(net, is_training=is_training)

        net = conv2d(net, 256, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=False, is_training=is_training,
                     scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=False, is_training=is_training,
                     scope='conv2', bn_decay=bn_decay)

    return net


def up_unit(x,
            up_ratio,
            upsample='nodeshuffle',
            k=16,
            idx=None,
            scope='up_block',
            use_att=False,
            **kwargs
            ):
    """
    The upsampling block
    :param x: sparse input
    :param up_ratio: x4 by default
    :param k:  number of neighbors
    :param scope: scope name
    :param upsample: the upsampling module used in the model.
    :return: upsampled points feature
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if upsample.lower() == 'nodeshuffle':
            x = nodeshuffle(x, up_ratio,
                            k=k,
                            idx=idx,
                            scope='nodeshuffle', **kwargs)

        elif upsample.lower() == 'duplicate':
            x = duplicate(x, up_ratio, scope='duplicate', **kwargs)

        elif upsample.lower() == 'multi_cnn':
            x = multi_cnn(x, up_ratio,
                          scope='multi_cnn', **kwargs)

        elif upsample.lower() == 'mlpshuffle':
            x = mlpshuffle(x, up_ratio,
                           scope='mlpshuffle', **kwargs)

        else:
            raise NotImplementedError('upsample type is not supported'.format(upsample))

        if use_att:
            x = attention_unit(x, **kwargs)

    return x


def down_block(inputs,
               up_ratio,
               scope='down_block', is_training=True, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = inputs
        net = tf.reshape(net, [tf.shape(net)[0], up_ratio, -1, tf.shape(net)[-1]])
        net = tf.transpose(net, [0, 2, 1, 3])

        net = conv2d(net, 256, [1, up_ratio],
                     padding='VALID', stride=[1, 1],
                     bn=False, is_training=is_training,
                     scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=False, is_training=is_training,
                     scope='conv2', bn_decay=bn_decay)

    return net


def feature_extraction(inputs,
                       growth_rate=12,
                       dense_n=3,
                       knn=16,
                       scope='feature_extraction2', is_training=True, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        use_bn = False
        use_ibn = False

        comp = growth_rate * 2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 24, [1, 1],
                             padding='VALID', scope='layer0',
                             is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)

        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer1", is_training=is_training, bn=use_bn, ibn=use_ibn,
                                         bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # (12+24*2)+24=84

        l2_features = conv1d(l1_features, comp, 1,  # 24
                             padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn,
                             bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 84+(24*2+12)=144

        l3_features = conv1d(l2_features, comp, 1,  # 48
                             padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn,
                             bn_decay=bn_decay)  # 48
        l3_features, l3_idx = dense_conv(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 144+(24*2+12)=204

        l4_features = conv1d(l3_features, comp, 1,  # 48
                             padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn,
                             bn_decay=bn_decay)  # 48
        l4_features, l3_idx = dense_conv(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 204+(24*2+12)=264

        l4_features = tf.expand_dims(l4_features, axis=2)

    return l4_features


# Feature Extraction Block used in point cloud upsanmpling network
# ------------------------------------------------------------
def feature_block(x,
                  block='inception',
                  k=16, d=2, growth_rate=12, n_dense=3,
                  use_global_pooling=True,
                  scope='FEB',
                  **kwargs):
    """
    Feature extraction block used in Dense Feature Extraction Module
    :param x: input features
    :param block: feature extraction block
    :param growth_rate: output channel growth rate
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path
    :param n_dense: number of layers in each DenseGCN block, default is 3
    :param scope: Feature extraction block
    :return: point feature
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # this is the default version of Inception DenseGCN in our paper PU-GCN
        if block == 'inception':
            y = inception_densegcn(x, growth_rate=growth_rate, k=k, d=d, n_dense=n_dense,
                                   use_global_pooling=use_global_pooling,
                                   scope='inception', **kwargs)

        # for ablating the residual connections
        # Inception DenseGCN without residual connection.
        elif block == 'inception_nores':
            y = inception_densegcn(x,
                                   growth_rate=growth_rate, k=k, d=d, n_dense=n_dense,
                                   use_global_pooling=use_global_pooling,
                                   use_residual=False,
                                   scope='inception_nores', **kwargs)

        # for ablating the dilated convolution
        # using different k in different path, no dilated GCN
        elif block == 'inception_nodil':
            y = inception_densegcn(x,
                                   growth_rate=growth_rate, k=k, d=d, n_dense=n_dense,
                                   use_global_pooling=use_global_pooling,
                                   use_dilation=False,
                                   scope='inception_res_densegcn_v0', **kwargs)
                                   # scope='inception_nodil', **kwargs)

        elif block == 'inception_1densegcn':
            y = vertex.inception_1densegcn(x, growth_rate=growth_rate, k=k, d=d, n_dense=n_dense,
                                           use_global_pooling=use_global_pooling,
                                           scope='inception_1densegcn', **kwargs)

        elif block == 'densegcn':
            # used in mpu and PU-GAN
            y = densegcn(x, growth_rate=growth_rate, n_layers=n_dense, k=k, d=d,
                         scope='densegcn', **kwargs)

        # for ablating the DenseGCN
        elif block == 'gcn':
            y = gcn(x, growth_rate=growth_rate, n_layers=n_dense, k=k, d=d,
                    scope='gcn', **kwargs)

        elif block == 'inceptiongcn':
            y = vertex.inceptiongcn(x, growth_rate=growth_rate, k=k, d=d, n_dense=n_dense,
                                    scope='inceptiongcn', **kwargs)


        else:
            raise NotImplementedError('{} is not implemented'.format(block))
        return y


# Feature Extractor used in point cloud upsampling network
# ----------------------------------------------- -------------
def feature_extractor(inputs,
                      block='inception', n_blocks=3,
                      growth_rate=24, k=16, d=2,
                      n_dense=3,
                      use_global_pooling=True,
                      scope='feature_extraction',
                      is_training=True, use_bn=False, use_ibn=False, bn_decay=None):
    """
    Dense Feature Extraction Module used in PU-GCN: Point Cloud upsampling using Graph Convolutional Networks.
    :param inputs: feature
    :param block: inception is the default one in PU-GCN
    :param n_blocks: number of feature extraction block inside the module.
    :param growth_rate: output channel of each path,
    :param k: the kernel size (num of neighbors)
    :param d: dilation rate
    :param n_dense: number of layers in each denseGCN block, default is 3
    :param scope:
    :param is_training:
    :param use_bn:
    :param use_ibn:
    :param bn_decay:
    :return: point features
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        l0_features = tf.expand_dims(inputs, axis=2)
        hidden_features, idx0 = densegcn(l0_features, growth_rate=growth_rate,
                                         n_layers=1, k=k, d=1,
                                         scope='head', return_idx=True,
                                         is_training=is_training, use_bn=use_bn, use_ibn=use_ibn, bn_decay=bn_decay)
        # feat_bank = []
        feat_fuse = 0.
        for i in range(0, n_blocks):
            hidden_features = feature_block(hidden_features,
                                            block=block,
                                            k=k, d=d, growth_rate=growth_rate, n_dense=n_dense,
                                            use_global_pooling=use_global_pooling,
                                            scope='FEB_' + str(i),
                                            is_training=is_training, bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn
                                            )
            feat_fuse += hidden_features
            # feat_bank.append(hidden_features)
        feat_fuse /= n_blocks
        # feature_fusion = tf.math.reduce_mean(tf.stack(feat_bank,  axis=0), axis=0, keepdims=False)
    return feat_fuse, idx0


def up_projection_unit(inputs,
                       up_ratio,
                       scope="up_projection_unit", is_training=True, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        L = conv2d(inputs, 128, [1, 1],
                   padding='VALID', stride=[1, 1],
                   bn=False, is_training=is_training,
                   scope='conv0', bn_decay=bn_decay)

        H0 = up_block(L, up_ratio, is_training=is_training, bn_decay=bn_decay, scope='up_0')

        L0 = down_block(H0, up_ratio, is_training=is_training, bn_decay=bn_decay, scope='down_0')
        E0 = L0 - L
        H1 = up_block(E0, up_ratio, is_training=is_training, bn_decay=bn_decay, scope='up_1')
        H2 = H0 + H1
    return H2


def coordinate_reconstruction_unit(inputs,
                                   scope="reconstruction", is_training=True, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        coord = conv2d(inputs, 32, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=False, is_training=is_training,
                       scope='fc_layer1', bn_decay=None)

        coord = conv2d(coord, 3, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=False, is_training=is_training,
                       scope='fc_layer2', bn_decay=None,
                       activation_fn=None, weight_decay=0.0)
        outputs = tf.squeeze(coord, [2])

        return outputs


def attention_unit(inputs, scope='attention_unit', is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape()[-1].value
        layer = dim // 4
        f = conv2d(inputs, layer, [1, 1],
                   padding='VALID', stride=[1, 1],
                   bn=False, is_training=is_training,
                   scope='conv_f', bn_decay=None)

        g = conv2d(inputs, layer, [1, 1],
                   padding='VALID', stride=[1, 1],
                   bn=False, is_training=is_training,
                   scope='conv_g', bn_decay=None)

        h = conv2d(inputs, dim, [1, 1],
                   padding='VALID', stride=[1, 1],
                   bn=False, is_training=is_training,
                   scope='conv_h', bn_decay=None)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = gamma * o + inputs

    return x

def encoder(inputs, scope='encoder', is_training=True):
    inputs = tf.expand_dims(inputs, 2)
    f1 = edgeformer(inputs, scope='1', is_training=is_training, dims=64)
    f2 = edgeformer(f1, scope='2', is_training=is_training, dims=64)
    f3 = edgeformer(f2, scope='3', is_training=is_training, dims=128)
    f4 = edgeformer(f3, scope='4', is_training=is_training, dims=256)
    
    feature = tf.expand_dims(tf.concat([f1, f2, f3, f4], axis=-1), 2)
    
    output = conv2d(feature, 512, [1, 1],
                    padding='VALID', stride=[1, 1],
                    bn=False, is_training=is_training,
                    scope='conv_final', bn_decay=None, activation_fn=None)
    
    return output

def edgeformer(inputs, scope='mhsa', is_training=True, dims=128, num_head=8):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        
        q = EdgeConv(inputs, C=dims, k=16, scope='conv_q', is_training=is_training, bn=False, bn_decay=None, activation=None)
        k = EdgeConv(inputs, C=dims, k=16, scope='conv_k', is_training=is_training, bn=False, bn_decay=None, activation=None)
        v = EdgeConv(inputs, C=dims, k=16, scope='conv_v', is_training=is_training, bn=False, bn_decay=None, activation=None)
        
        indices = dims//num_head
                
        # attention unit
        for i in range(num_head):
            query = q[:,:,i*indices:(i+1)*indices-1]
            key = k[:,:,i*indices:(i+1)*indices-1]
            value = v[:,:,i*indices:(i+1)*indices-1]
            
            # 1. MatMul
            matmul_qk = tf.matmul(query, key, transpose_b=True)

            # 2. Scale
            #dk = tf.cast(tf.shape(k)[-1], tf.float32)
            #scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            scaled_attention_logits = matmul_qk

            # 3. Mask (opt.)
            #if mask is not None:
            #    scaled_attention_logits += (mask * -1e9)

            # 4. SoftMax      
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

            # 5. MatMul
            output = tf.matmul(attention_weights, value)
            
            if i==0:
                net = output
            else:
                net = tf.concat([net, output], axis=-1)
        
        output = EdgeConv(net, C=dims, k=8, scope='conv_o', is_training=is_training, bn=False, bn_decay=None, activation=tf.nn.relu)
                
    return output

##################################################################################
# Other function
##################################################################################
def instance_norm(net, train=True, weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift', shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device("/cpu:0"):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def nnconv1d(inputs,
           num_output_channels,
           kernel_size=1,
           scope="conv1d",
           stride=1,
           padding='VALID',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    assert(data_format=='NHWC' or data_format=='NCHW')
    if data_format == 'NHWC':
      num_in_channels = inputs.get_shape()[-1].value
    elif data_format=='NCHW':
      num_in_channels = inputs.get_shape()[1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding,
                           data_format=data_format)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn',
                                      data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
                
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size=[1, 1],
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=tf.AUTO_REUSE):
    """ 2D convolution with non-linear operation.

    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv2d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False,
                                                    fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias=True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs, num_outputs,
                                  use_bias=use_bias, kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k + 1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx


def dense_conv(feature, n=3, growth_rate=64, k=16, scope='dense_conv', **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n - 1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y, idx


def nn_dense_conv(feature, n=3,growth_rate=64, k=16, scope='dense_conv',**kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    tf.nn.relu(conv2d(y, growth_rate, name='l%d' % i)),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, name='l%d' % i),
                    y], axis=-1)
            else:

                y = tf.concat([
                    tf.nn.relu(conv2d(y, growth_rate, name='l%d' % i)),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y,idx

def EdgeConv(inputs, C=64, k=16, scope=None, is_training=None, bn=False, bn_decay=None, activation=tf.nn.relu):
  '''
    EdgeConv layer:
      Wang, Y, Yongbin S, Ziwei L, Sanjay S, Michael B, Justin S.
      "Dynamic graph cnn for learning on point clouds."
      arXiv:1801.07829 (2018).
  '''

  adj_matrix = pairwise_distance(inputs)
  nn_idx = knn(adj_matrix, k=k)

  edge_features = nn_get_edge_feature(inputs, nn_idx, k)
  out = conv2d(edge_features, C,
               bn=bn, is_training=is_training,
               scope=scope, bn_decay=bn_decay,
               activation_fn=activation)
  vertex_features = tf.reduce_max(out, axis=-2)

  return vertex_features


def normalize_point_cloud(pc):
    """
    pc [N, P, 3]
    """
    centroid = tf.reduce_mean(pc, axis=1, keep_dims=True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keep_dims=True)), axis=1, keep_dims=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance


def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


def safe_log(x, eps=1e-12):
    return tf.log(x + eps)


def tf_covariance(data):
    ## x: [batch_size, num_point, k, 3]
    batch_size = data.get_shape()[0].value
    num_point = data.get_shape()[1].value

    mean_data = tf.reduce_mean(data, axis=2, keep_dims=True)  # (batch_size, num_point, 1, 3)
    mx = tf.matmul(tf.transpose(mean_data, perm=[0, 1, 3, 2]), mean_data)  # (batch_size, num_point, 3, 3)
    vx = tf.matmul(tf.transpose(data, perm=[0, 1, 3, 2]), data) / tf.cast(tf.shape(data)[0],
                                                                          tf.float32)  # (batch_size, num_point, 3, 3)
    data_cov = tf.reshape(vx - mx, shape=[batch_size, num_point, -1])

    return data_cov


def add_scalar_summary(name, value, collection='train_summary'):
    tf.summary.scalar(name, value, collections=[collection])


def add_hist_summary(name, value, collection='train_summary'):
    tf.summary.histogram(name, value, collections=[collection])


def add_train_scalar_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])


def add_train_hist_summary(name, value):
    tf.summary.histogram(name, value, collections=['train_summary'])


def add_train_image_summary(name, value):
    tf.summary.image(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update

def nn_get_edge_feature(point_cloud, nn_idx, k):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_central = point_cloud

  point_cloud_shape = point_cloud.get_shape()
  batch_size = point_cloud_shape[0].value
  num_points = point_cloud_shape[1].value
  num_dims = point_cloud_shape[2].value

  idx_ = tf.range(batch_size) * num_points
  idx_ = tf.reshape(idx_, [batch_size, 1, 1])

  point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
  point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
  point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

  point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

  edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
  return edge_feature

def pairwise_distance(point_cloud):
  """Compute pairwise distance of a point cloud.

  Args:
    point_cloud: tensor (batch_size, num_points, num_dims)

  Returns:
    pairwise distance: (batch_size, num_points, num_points)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
  point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
  point_cloud_inner = -2*point_cloud_inner
  point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
  point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
  return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def knn(adj_matrix, k=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int

  Returns:
    nearest neighbors: (batch_size, num_points, k)
  """
  neg_adj = -adj_matrix
  _, nn_idx = tf.nn.top_k(neg_adj, k=k)
  return nn_idx