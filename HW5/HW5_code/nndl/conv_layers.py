import numpy as np
from nndl.layers import *
import pdb


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

  N, C, H, W = x.shape #get the shape of x
  F, C, HH, WW = w.shape #get the shape of w

  #calculate the output size
  Hout = int(1 + (H + 2 * pad - HH) / stride)
  Wout = int(1 + (W + 2 * pad - WW) / stride)

  #x pad
  padx = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode = 'constant')

  #make the output to be the correct size
  out = np.zeros([N, F, Hout, Wout])

  #compute the out
  for data in np.arange(N): #for each data point
    for filter in np.arange(F): #for each filter
      for height in np.arange(Hout): #for each row
        for width in np.arange(Wout): #for each column
            out[data, filter, height, width] = np.sum(padx[data, :, height * stride : height * stride + HH, width * stride : width * stride + WW] * w[filter, :, :, :]) + b[filter]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache

  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #

  N, C, H, W = x.shape #get the shape of x
  F, C, HH, WW = w.shape #get the shape of w

  #make the output to be the correct size
  dxpad = np.zeros(xpad.shape)
  dx = np.zeros(x.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)

  #calculate db
  for filter in np.arange(F):
      db[filter] += np.sum(dout[:, filter, :, :])  # sum all data point's filters

  for data in np.arange(N): #for each data point
      for filter in np.arange(F): #for each filter
          for height in np.arange(out_height): #for each row
              for width in np.arange(out_width): #for each column
                  #dw = xpad[] * dout
                  dw[filter] += xpad[data, :, height * stride : height * stride + HH, width * stride : width * stride + WW] * dout[data, filter, height, width]
                  #dx = w * dout
                  dxpad[data, :, height * stride : height * stride + HH, width * stride : width * stride + WW] += w[filter] * dout[data, filter, height, width]

  #update dx
  dx = dxpad[:, :, pad : -pad, pad : -pad]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #

  #get the params of pool filter
  pool_height = pool_param.get('pool_height')
  pool_width = pool_param.get('pool_width')
  stride = pool_param.get('stride')

  N, C, H, W = x.shape #get the shape of x

  #calculate the output size
  Hout = int(1 + (H - pool_height) / stride)
  Wout = int(1 + (W - pool_width) / stride)

  #make the output to be the correct size
  out = np.zeros([N, C, Hout, Wout])

  for data in np.arange(N): # for each data point
    for channel in np.arange(C): # for each channel
        for height in np.arange(Hout): #for each row
            for width in np.arange(Wout): #for each column
                out[data, channel, height, width] = np.max(x[data, channel, height * stride : height * stride + pool_height, width * stride : width * stride + pool_width])


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #

  N, C, H, W = x.shape #get the shape of x
  _, _, dout_height, dout_width = dout.shape #get the shape of dout

  #make the output to be the correct size
  dx = np.zeros(x.shape)

  for data in np.arange(N): #for each data point
    for channel in np.arange(C): #for each channel
        for height in np.arange(dout_height): #for each row
            for width in np.arange(dout_width): #for each column
                maxnum = np.argmax(x[data, channel, height * stride : height * stride + pool_height, width * stride : width * stride + pool_width])
                maxfield = np.unravel_index(maxnum, [pool_height, pool_width])
                dx[data, channel, height * stride : height * stride + pool_height, width * stride : width * stride + pool_width][maxfield] = dout[data, channel, height, width]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you
  #   implemented in HW #4.
  # ================================================================ #

  N, C, H, W = x.shape #get the shape of x

  #reshape the input to 2D array for Batch Normalization
  x = x.reshape(N, H, W, C)
  x = x.reshape(N * H * W, C)

  #do the batchnorm forward
  out, cache = batchnorm_forward(x, gamma, beta, bn_param)

  #reshape the output to (N, C, H, W)
  out = out.reshape((N, H, W, C)).transpose(0, 3, 1, 2)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you
  #   implemented in HW #4.
  # ================================================================ #

  N, C, H, W = dout.shape #get the shape of dout

  #reshape the dout
  dout = (dout.transpose(0, 2, 3, 1)).reshape(N * H * W, C)

  #do the batchnorm backward
  dx, dgamma, dbeta = batchnorm_backward(dout, cache)

  #reshape the output to de desired format
  dx = dx.reshape(N, C, H,W)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dgamma, dbeta
