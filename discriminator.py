"""
    The ``model`` module
    ======================
 
    Contains the class Model which implements the core model for CG detection, 
    training, testing and visualization functions.
"""

import os

import os.path 
import time
import random
from . import image_loader as il
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import csv
import configparser

import numpy as np

from PIL import Image

GPU = '/gpu:0'
config = 'server'

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score as acc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

import pickle

# seed initialisation
print("\n   random initialisation ...")
random_seed = int(time.time() % 10000 ) 
random.seed(random_seed)  # for reproducibility
print('   random seed =', random_seed)


"""
nb_class
nl
nf
filter_size
feature_extractor
"""

def weight_variable(shape, nb_input, seed = None):
  """Creates and initializes (truncated normal distribution) a variable weight Tensor with a defined shape"""
  sigma = np.sqrt(2/nb_input)
  # print(sigma)
  initial = tf.truncated_normal(shape, stddev=sigma, seed = random_seed)
  return tf.Variable(initial)

def bias_variable(shape):
  """Creates and initializes (truncated normal distribution with 0.5 mean) a variable bias Tensor with a defined shape"""
  initial = tf.truncated_normal(shape, mean = 0.5, stddev=0.1, seed = random_seed)
  return tf.Variable(initial)
  
def conv2d(x, W):
  """Returns the 2D convolution between input x and the kernel W"""  
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x):
  """Returns the result of max-pooling on input x with a 2x2 window""" 
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
  """Returns the result of average-pooling on input x with a 2x2 window""" 
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

def max_pool_10x10(x):
  """Returns the result of max-pooling on input x with a 10x10 window""" 
  return tf.nn.max_pool(x, ksize=[1, 10, 10, 1],
                           strides=[1, 10, 10, 1], padding='SAME')

def avg_pool_10x10(x):
  """Returns the result of average-pooling on input x with a 10x10 window""" 
  return tf.nn.avg_pool(x, ksize=[1, 10, 10, 1],
                           strides=[1, 10, 10, 1], padding='SAME')

def histogram(x, nbins):
  """Returns the Tensor containing the nbins values of the normalized histogram of x""" 
  h = tf.histogram_fixed_width(x, value_range = [-1.0,1.0], 
                               nbins = nbins, dtype = tf.float32)
  return(h)

def gaussian_func(mu, x, n, sigma):
  """Returns the average of x composed with a gaussian function

    :param mu: The mean of the gaussian function
    :param x: Input values 
    :param n: Number of input values
    :param sigma: Variance of the gaussian function
    :type mu: float
    :type x: Tensor
    :type n: int 
    :type sigma: float
  """ 
  gauss = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
  # return(tf.reduce_sum(gauss.pdf(xmax - tf.nn.relu(xmax - x))/n))
  return(tf.reduce_sum(gauss.pdf(x)/n))



def gaussian_kernel(x, nbins = 8, values_range = [0, 1], sigma = 0.1,image_size = 100):
  """Returns the values of x's nbins gaussian histogram 

    :param x: Input values (supposed to be images)
    :param nbins: Number of bins (different gaussian kernels)
    :param values_range: The range of the x values
    :param sigma: Variance of the gaussian functions
    :param image_size: The size of the images x (for normalization)
    :type x: Tensor
    :type nbins: int 
    :type values_range: table
    :type sigma: float
    :type image_size: int
  """ 
  mu_list = np.float32(np.linspace(values_range[0], values_range[1], nbins + 1))
  n = np.float32(image_size**2)
  function_to_map = lambda m : gaussian_func(m, x, n, sigma)
  return(tf.map_fn(function_to_map, mu_list))

def plot_gaussian_kernel(nbins = 8, values_range = [0, 1], sigma = 0.1):
  """Plots the gaussian kernels used for estimating the histogram"""

  r = values_range[1] - values_range[0]
  mu_list = []
  for i in range(nbins+1):
    mu_list.append(values_range[0] + i*r/(nbins+1))

  range_plot = np.linspace(values_range[0]-0.1, values_range[1]+0.1, 1000)

  plt.figure()
  for mu in mu_list:
    plt.plot(range_plot, np.exp(-(range_plot-mu)**2/(sigma**2)))
  plt.title("Gaussian kernels used for estimating the histograms")
  plt.show()


def classic_histogram_gaussian(x, k, nbins = 8, values_range = [0, 1], sigma = 0.6):
  """Computes gaussian histogram values for k input images"""
  function_to_map = lambda y: tf.stack([gaussian_kernel(y[:,:,i], nbins, values_range, sigma) for i in range(k)])
  res = tf.map_fn(function_to_map, x)
  return(res)

def stat(x):
  """Computes statistical features for an image x : mean, min, max and variance"""
  # sigma = tf.reduce_mean((x - tf.reduce_mean(x))**2)
  return(tf.stack([tf.reduce_mean(x), tf.reduce_min(x), tf.reduce_max(x), tf.reduce_mean((x - tf.reduce_mean(x))**2)]))

def compute_stat(x, k):
  """Computes statistical features for k images"""
  # function_to_map = lambda y: tf.stack([stat(y[:,:,i]) for i in range(k)])
  # res = tf.map_fn(function_to_map, x)
  res = tf.transpose(tf.stack([tf.reduce_mean(x, axis=[1,2]), tf.reduce_min(x, axis=[1,2]), tf.reduce_max(x, axis=[1,2]), tf.reduce_mean((x - tf.reduce_mean(x, axis=[1,2], keep_dims = True))**2, axis=[1,2])]), [1,2,0])
  return(res)


  
def create_graph(nb_class, nl = 2, nf = [32, 64], filter_size = 3,
                   feature_extractor = 'Stats'): 
    """Creates the TensorFlow graph"""

    print('   create model ...')
    # input layer. One entry is a float size x size, 3-channels image. 
    # None means that the number of such vector can be of any lenght.

    if feature_extractor == 'Hist': 
      print('   Model with histograms.')

    else: 
      print('   Model with statistics.')

    graph = tf.Graph()

    with graph.as_default():

      with tf.name_scope('Input_Data'):
        x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.nb_channels])
        self.x = x
        # reshape the input data:
        x_image = tf.reshape(x, [-1,self.image_size, self.image_size, self.nb_channels])
        with tf.name_scope('Image_Visualization'):
          tf.summary.image('Input_Data', x_image)
        

      # first conv net layer
      if self.remove_context:
        print('   Creating layer 1 - Shape : ' + str(self.remove_filter_size) + 'x' + 
              str(self.remove_filter_size) + 'x' + str(self.nb_channels) + 'x' + str(nf[0]))
      else:
        print('   Creating layer 1 - Shape : ' + str(self.filter_size) + 'x' + 
              str(self.filter_size) + 'x' + str(self.nb_channels) + 'x' + str(nf[0]))      

      with tf.name_scope('Conv1'):

        with tf.name_scope('Weights'):
          if self.remove_context:
            W_conv1 = weight_variable([self.remove_filter_size, self.remove_filter_size, self.nb_channels, nf[0]], 
                                      nb_input = self.remove_filter_size*self.remove_filter_size*self.nb_channels,
                                      seed = random_seed)
          else:
            W_conv1 = weight_variable([self.filter_size, self.filter_size, self.nb_channels, nf[0]], 
                                      nb_input = self.filter_size*self.filter_size*self.nb_channels,
                                      seed = random_seed)
          self.W_conv1 = W_conv1
        with tf.name_scope('Bias'):
          b_conv1 = bias_variable([nf[0]])


        # relu on the conv layer
        if self.remove_context: 
          h_conv1 = conv2d(x_image, W_conv1)
        else:         
          h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, 
                               name = 'Activated_1')
        self.h_conv1 = h_conv1

      self.W_convs = [W_conv1]
      self.b_convs = [b_conv1]
      self.h_convs = [h_conv1]

      image_summaries(self.h_convs[0], 'hconv1')
      filter_summary(self.W_convs[0], 'Wconv1')

      for i in range(1, nl):
        print('   Creating layer ' + str(i+1) + ' - Shape : ' + str(self.filter_size) + 'x' + 
            str(self.filter_size) + 'x' + str(nf[i-1]) + 'x' + str(nf[i]))
        # other conv 
        with tf.name_scope('Conv' + str(i+1)):
          with tf.name_scope('Weights'):
            W_conv2 = weight_variable([self.filter_size, self.filter_size, nf[i-1], nf[i]],
                                      self.filter_size*self.filter_size*nf[i-1])
            self.W_convs.append(W_conv2)
          with tf.name_scope('Bias'):
            b_conv2 = bias_variable([nf[i]])
            self.b_convs.append(b_conv2)

          h_conv2 = tf.nn.relu(conv2d(self.h_convs[i-1], W_conv2) + b_conv2, 
                               name = 'Activated_2')

          self.h_convs.append(h_conv2)    


      print('   Creating feature extraction layer')
      nb_filters = nf[nl-1]
      if self.feature_extractor == 'Hist':
        # Histograms
        nbins = self.nbins
        size_flat = (nbins + 1)*nb_filters

        range_hist = [0,1]
        sigma = 0.07

        # plot_gaussian_kernel(nbins = nbins, values_range = range_hist, sigma = sigma)

        with tf.name_scope('Gaussian_Histogram'): 
          hist = classic_histogram_gaussian(self.h_convs[nl-1], k = nb_filters, 
                                            nbins = nbins, 
                                            values_range = range_hist, 
                                            sigma = sigma)
          self.hist = hist

        flatten = tf.reshape(hist, [-1, size_flat], name = "Flatten_Hist")
        self.flatten = flatten

      else: 
        nb_stats = 4
        size_flat = nb_filters*nb_stats
        with tf.name_scope('Simple_statistics'): 
          s = compute_stat(self.h_convs[nl-1], nb_filters)
          self.stat = s
          
        flatten = tf.reshape(s, [-1, size_flat], name = "Flattened_Stat")
        self.flatten = flatten


      print('   Creating MLP ')
      # Densely Connected Layer
      # we add a fully-connected layer with 1024 neurons 
      with tf.variable_scope('Dense1'):
        with tf.name_scope('Weights'):
          W_fc1 = weight_variable([size_flat, 1024],
                                  nb_input = size_flat)
        with tf.name_scope('Bias'):
          b_fc1 = bias_variable([1024])
        # put a relu
        h_fc1 = tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1, 
                           name = 'activated')

      # dropout
      with tf.name_scope('Dropout1'):
        keep_prob = tf.placeholder(tf.float32)
        self.keep_prob = keep_prob
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      self.h_fc1 = h_fc1

      # readout layer
      with tf.variable_scope('Readout'):
        with tf.name_scope('Weights'):
          W_fc3 = weight_variable([1024, nb_class],
                                  nb_input = 1024)
        with tf.name_scope('Bias'):
          b_fc3 = bias_variable([nb_class])
        y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

      self.y_conv = y_conv

      # support for the learning label
      y_ = tf.placeholder(tf.float32, [None, nb_class])
      self.y_ = y_



      # Define loss (cost) function and optimizer
      print('   setup loss function and optimizer ...')

      # softmax to have normalized class probabilities + cross-entropy
      with tf.name_scope('cross_entropy'):

        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)
        with tf.name_scope('total'):
          cross_entropy_mean = tf.reduce_mean(softmax_cross_entropy)

      tf.summary.scalar('cross_entropy', cross_entropy_mean)

      with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean)

      # with tf.name_scope('enforce_constraints'):
      if self.remove_context:
        # self.zero_op = tf.assign(ref = self.W_convs[0][1,1,0,:], value = tf.zeros([nf[0]]))
        center = int(self.remove_filter_size/2)
        self.zero_op = tf.scatter_nd_update(ref = self.W_convs[0], indices = tf.constant([[center,center,0,i] for i in range(nf[0])]), updates = tf.zeros(nf[0]))
        self.norm_op = tf.assign(ref = self.W_convs[0], value = tf.divide(self.W_convs[0],tf.reduce_sum(self.W_convs[0], axis = 3, keep_dims = True)))
        self.minus_one_op = tf.scatter_nd_update(ref = self.W_convs[0], indices = tf.constant([[center,center,0,i] for i in range(nf[0])]), updates = tf.constant([-1.0 for i in range(nf[0])]))
        self.norm = tf.reduce_sum(self.W_convs[0], axis = 3, keep_dims = True)

      self.train_step = train_step
      print('   test ...')
      # 'correct_prediction' is a function. argmax(y, 1), here 1 is for the axis number 1
      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

      # 'accuracy' is a function: cast the boolean prediction to float and average them
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

      self.accuracy = accuracy

    self.graph = graph
    print('   model created.')