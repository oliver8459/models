# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow.compat.v1 as tf
import tf_slim as slim

# Conv and DepthSepConv namedtuple define layers of the shufflenet_v2 architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
MaxPool = namedtuple('max_pool2d', ['kernel', 'stride', 'padding'])
ShuffleBlock = namedtuple('shufflenet_v2_block', ['kernel', 'stride', 'depth'])

stage_depth = [48, 96, 192] # 0.5x
# stage_depth = [116, 232, 464] # 1.0x


# SHUFFLENETV2_CONV_DEFS specifies the shufflenet_v2 body
SHUFFLENETV2_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=24), # 0
    MaxPool(kernel=3, stride=2, padding='SAME'),  # 1
    # 4
    ShuffleBlock(kernel=[3, 3], stride=2, depth=stage_depth[0]), # 2
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[0]), # 3
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[0]), # 4
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[0]), # 5
    # 8
    ShuffleBlock(kernel=[3, 3], stride=2, depth=stage_depth[1]), # 6
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[1]), # 7
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[1]), # 8
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[1]), # 9
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[1]), # 10
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[1]), # 11
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[1]), # 12
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[1]), # 13
    # 4
    ShuffleBlock(kernel=[3, 3], stride=2, depth=stage_depth[2]), # 14
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[2]), # 15
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[2]), # 16
    ShuffleBlock(kernel=[3, 3], stride=1, depth=stage_depth[2])  # 17
]


def _fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  Pads the input such that if it was used in a convolution with 'VALID' padding,
  the output would have the same dimensions as if the unpadded input was used
  in a convolution with 'SAME' padding.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                           kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
  pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
  pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
  pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
  padded_inputs = tf.pad(
      tensor=inputs,
      paddings=[[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]],
                [0, 0]])
  return padded_inputs


def shufflenet_v2_base(inputs,
                      final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      use_explicit_padding=False,
                      scope=None):
  """shufflenet v2.

  Constructs a shufflenet v2 network from inputs to the given final endpoint.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
      'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5'_pointwise,
      'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
      'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
      'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0, or the target output_stride is not
                allowed.
  """
  depth = lambda d: max(int(d * depth_multiplier), min_depth)
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  conv_defs = SHUFFLENETV2_CONV_DEFS

  if output_stride is not None and output_stride not in [8, 16, 32]:
    raise ValueError('Only allowed output_stride values are 8, 16, 32.')

  padding = 'SAME'
  if use_explicit_padding:
    padding = 'VALID'
  with tf.variable_scope(scope, 'ShufflenetV2', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
      # The current_stride variable keeps track of the output stride of the
      # activations, i.e., the running product of convolution strides up to the
      # current network layer. This allows us to invoke atrous convolution
      # whenever applying the next convolution would result in the activations
      # having output stride larger than the target output_stride.
      current_stride = 1

      # The atrous convolution rate parameter.
      rate = 1

      net = inputs
      for i, conv_def in enumerate(conv_defs):
        end_point_base = 'Conv2d_%d' % i

        if output_stride is not None and current_stride == output_stride:
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          layer_stride = 1
          layer_rate = rate
          rate *= conv_def.stride
        else:
          layer_stride = conv_def.stride
          layer_rate = 1
          current_stride *= conv_def.stride

        if isinstance(conv_def, Conv):
          end_point = end_point_base
          if use_explicit_padding:
            net = _fixed_padding(net, conv_def.kernel)
          net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                            stride=conv_def.stride,
                            scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points

        elif isinstance(conv_def, DepthSepConv):
          end_point = end_point_base + '_depthwise'

          # By passing filters=None
          # separable_conv2d produces only a depthwise convolution layer
          if use_explicit_padding:
            net = _fixed_padding(net, conv_def.kernel, layer_rate)
          net = slim.separable_conv2d(net, None, conv_def.kernel,
                                      depth_multiplier=1,
                                      stride=layer_stride,
                                      rate=layer_rate,
                                      scope=end_point)

          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points

          end_point = end_point_base + '_pointwise'

          net = slim.conv2d(net, depth(conv_def.depth), [1, 1],
                            stride=1,
                            scope=end_point)

          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points

        elif isinstance(conv_def, MaxPool):
            end_point = end_point_base + '_maxpool'

            # By passing filters=None
            # separable_conv2d produces only a depthwise convolution layer
            if use_explicit_padding:
                net = _fixed_padding(net, conv_def.kernel, layer_rate)
            net = slim.max_pool2d(net, kernel_size=conv_def.kernel, stride=conv_def.stride, padding=conv_def.padding)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

        elif isinstance(conv_def, ShuffleBlock):
            end_point = end_point_base + '_shuffleblock'

            # By passing filters=None
            # separable_conv2d produces only a depthwise convolution layer
            if use_explicit_padding:
                net = _fixed_padding(net, conv_def.kernel, layer_rate)
            net = shufflenet_v2_block(net, out_channel=conv_def.depth,
                                      kernel_size=conv_def.kernel, stride=conv_def.stride)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points
        else:
          raise ValueError('Unknown convolution type %s for layer %d'
                           % (conv_def.ltype, i))
  raise ValueError('Unknown final endpoint %s' % final_endpoint)


def shufflenet_v2(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='ShufflenetV2',
                 global_pool=False):
  """shufflenet_v2 model for classification.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.

  Returns:
    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: Input rank is invalid.
  """
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))

  with tf.variable_scope(
      scope, 'ShufflenetV2', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = shufflenet_v2_base(inputs, scope=scope,
                                          min_depth=min_depth,
                                          depth_multiplier=depth_multiplier,
                                          conv_defs=conv_defs)
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(
              input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a')
          end_points['AvgPool_1a'] = net
        if not num_classes:
          return net, end_points
        # 1 x 1 x 1024
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      if prediction_fn:
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points

shufflenet_v2.default_image_size = 224


def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func


shufflenet_v2_075 = wrapped_partial(shufflenet_v2, depth_multiplier=0.75)
shufflenet_v2_050 = wrapped_partial(shufflenet_v2, depth_multiplier=0.50)
shufflenet_v2_025 = wrapped_partial(shufflenet_v2, depth_multiplier=0.25)


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def shufflenet_v2_arg_scope(
    is_training=True,
    weight_decay=0.00004,
    stddev=0.09,
    regularize_depthwise=False,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
    normalizer_fn=slim.batch_norm):
  """Defines the default shufflenet_v2 arg scope.

  Args:
    is_training: Whether or not we're training the model. If this is set to
      None, the parameter is not added to the batch_norm arg_scope.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
    normalizer_fn: Normalization function to apply after convolution.

  Returns:
    An `arg_scope` to use for the shufflenet v1 model.
  """
  batch_norm_params = {
      'center': True,
      'scale': True,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'updates_collections': batch_norm_updates_collections,
  }
  if is_training is not None:
    batch_norm_params['is_training'] = is_training

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = slim.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=normalizer_fn):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc


def shufflenet_v2_block(x, out_channel, kernel_size, stride=1, dilation=1, shuffle_group=2, name=None):
    if stride == 1:
        top, bottom = tf.split(x, num_or_size_splits=2, axis=3)
        half_channel = out_channel // 2

        top = slim.conv2d(top, half_channel, kernel_size=1, stride=stride)
        top = slim.separable_conv2d(top, None, kernel_size=kernel_size, depth_multiplier=1, stride=stride,
                                  rate=dilation)
        top = slim.conv2d(top, half_channel, kernel_size=1, stride=stride)

        out = tf.concat([top, bottom], axis=3)
        out = shuffle_unit(out, shuffle_group, name)

    else:
        half_channel = out_channel // 2
        b0 = slim.conv2d(x, half_channel, kernel_size=1, stride=1)
        b0 = slim.separable_conv2d(b0, None, kernel_size=kernel_size, depth_multiplier=1, stride=stride,
                                    rate=dilation)
        b0 = slim.conv2d(b0, half_channel, kernel_size=1, stride=1)

        b1 = slim.separable_conv2d(x, None, kernel_size=kernel_size, depth_multiplier=1, stride=stride,
                                   rate=dilation)
        b1 = slim.conv2d(b1, half_channel, kernel_size=1, stride=1)

        out = tf.concat([b0, b1], axis=3)
        out = shuffle_unit(out, shuffle_group, name)
    return out


# 不会用到tf.split 但是loss不下降 
# def shuffle_unit(x, groups, name):
#     n, h, w, c = x.get_shape().as_list()
#     x = tf.reshape(x, shape=tf.convert_to_tensor([n, h, w, groups, c//groups]))
#     x = tf.transpose(x, tf.convert_to_tensor([3, 0, 1, 2, 4]), name=name)
#     return x[0], x[1]

# 会用到tf.split tensorflow1.13转tflite不能用
def shuffle_unit(x, groups, name):
    n, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, shape=tf.convert_to_tensor([n, h, w, groups, c//groups]))
    x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]), name=name)
    x = tf.reshape(x, shape=tf.convert_to_tensor([n, h, w, c]))
    return x

"""
ShuffleNet contains 5-D tensor.If need to convert ot tflite model, 
squeezing the first channel to 4-D is neccessary before exporting to inference graph

usage: use new shuffle_unit() replace the older one, and recompile proto
cd ${models}/research && protoc object_detection/protos/*.proto --python_out=.
"""
# 不会用到tf.split 但是loss不下降 
# def shuffle_unit(x, groups, name):
#     n, h, w, c = x.get_shape().as_list()
#     x = tf.squeeze(x)
#     x = tf.reshape(x, shape=tf.convert_to_tensor([h, w, groups, c//groups]))
#     x = tf.transpose(x, tf.convert_to_tensor([2, 0, 1, 3]))
#     return tf.expand_dims(x[0], axis=0, name=name), tf.expand_dims(x[1], axis=0, name=name)

# 会用到tf.split tensorflow1.13转tflite不能用
# def shuffle_unit(x, groups, name):
#     n, h, w, c = x.get_shape().as_list()
#     x = tf.squeeze(x)
#     x = tf.reshape(x, shape=tf.convert_to_tensor([h, w, groups, c//groups]))
#     x = tf.transpose(x, tf.convert_to_tensor([0, 1, 3, 2]), name=name)
#     x = tf.reshape(x, shape=tf.convert_to_tensor([n, h, w, c]))
#     return x
