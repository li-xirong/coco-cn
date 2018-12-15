from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

class MLP(object):

  def __init__(self, is_training, n_layers, keep_prob, last_nl_fun, loss_fun, top_k, aux_classes=0, flag_with_saver=False, gpu=1):
    nb_feature = n_layers[0]
    nb_classes = n_layers[-1]

    if not is_training:
        keep_prob = 1.0


    # Create a multilayer model.
    with tf.device('/gpu:%d' % gpu):
      # Input placeholders
      with tf.name_scope('input'):
        self._input_data = x = tf.placeholder(tf.float32, [None, nb_feature], name='x-input')
        self._targets = y_ = tf.placeholder(tf.float32, [None, nb_classes], name='y-input')
        self._aux_targets = y2_ = tf.placeholder(tf.float32, [None, aux_classes], name='y2-input') if aux_classes else None

      # We can't initialize these variables to 0 - the network will get stuck.
      def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

      def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

      def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
          # This Variable will hold the state of the weights for the layer
          with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
          with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
          with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
          if act is not None:
            activations = act(preactivate, name='activation')
            return activations
          else:
            return preactivate


      if len(n_layers) == 3:
        hidden1 = nn_layer(x, n_layers[0], n_layers[1], 'layer1')
        with tf.name_scope('dropout'):
          #self._keep_prob = keep_prob = tf.placeholder(tf.float32)
          dropped = tf.nn.dropout(hidden1, keep_prob)

      elif len(n_layers) == 4:
        hidden1 = nn_layer(x, n_layers[0], n_layers[1], 'hidden_layer_1')
        with tf.name_scope('dropout'):
          #self._keep_prob = keep_prob = tf.placeholder(tf.float32)
          dropped1 = tf.nn.dropout(hidden1, keep_prob)

        hidden2 = nn_layer(dropped1, n_layers[1], n_layers[2], 'hidden_layer_2')
        with tf.name_scope('dropout'):
          # keep_prob = tf.placeholder(tf.float32)
          dropped = tf.nn.dropout(hidden2, keep_prob) 


      elif len(n_layers) == 5:
        hidden1 = nn_layer(x, n_layers[0], n_layers[1], 'hidden_layer_1')
        with tf.name_scope('dropout'):
          self._keep_prob = keep_prob = tf.placeholder(tf.float32)
          dropped1 = tf.nn.dropout(hidden1, keep_prob)

        hidden2 = nn_layer(dropped1, n_layers[1], n_layers[2], 'hidden_layer_2')
        with tf.name_scope('dropout'):
          # keep_prob = tf.placeholder(tf.float32)
          dropped2 = tf.nn.dropout(hidden2, keep_prob) 

        hidden3 = nn_layer(dropped2, n_layers[2], n_layers[3], 'hidden_layer_3')
        with tf.name_scope('dropout'):
          # keep_prob = tf.placeholder(tf.float32)
          dropped = tf.nn.dropout(hidden3, keep_prob) 


      self._logit = logit = nn_layer(dropped, n_layers[-2], n_layers[-1], 'layer2', act=None)
      if aux_classes > 0:
          self._aux_logit = aux_logit = nn_layer(dropped, n_layers[-2], aux_classes, 'layer3', act=None)
      if last_nl_fun == 'softmax':
        self._outputs = y = tf.nn.softmax(logit)
        if aux_classes > 0:
            self._aux_outputs = y2 = tf.nn.softmax(aux_logit)
        if loss_fun == 'cross_entropy':
          with tf.name_scope('cross_entropy'):
            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, y_))
            if aux_classes > 0:
                self._loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(aux_logit, y2_))
        if loss_fun == 'logistic':
          # logistic loss function refers to CVPR 2016 paper What Value Do Explicit High Level Concepts Have in Vision to Language Problems
          with tf.name_scope('logistic'):
            self._loss = tf.reduce_mean(tf.reduce_sum(tf.log(1+tf.exp(-y*y_)),1))
            if en_classes > 0:
                self._loss += tf.reduce_mean(tf.reduce_sum(tf.log(1+tf.exp(-y2*y2_)),1))
        

      elif last_nl_fun == 'sigmoid':
        self._outputs = y = tf.sigmoid(logit)
        if aux_classes > 0:
            self._aux_outputs = y2 = tf.sigmoid(aux_logit)
        if loss_fun == 'cross_entropy':
          with tf.name_scope('cross_entropy'):
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y_))
            if aux_classes > 0:
                self._loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=aux_logit, labels=y2_))
        if loss_fun == 'logistic':
          with tf.name_scope('logistic'):
            self._loss = tf.reduce_mean(tf.reduce_sum(tf.log(1+tf.exp(-y*y_)),1))
            if en_classes > 0:
                self._loss += tf.reduce_mean(tf.reduce_sum(tf.log(1+tf.exp(-y2*y2_)),1))



      with tf.name_scope('accuracy'):
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y*y_, 1))
        self._accuracy = accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self._pred_top_values, self._pred_top_indices = tf.nn.top_k(y, k=top_k)
        #self._target_top_values, self._target_top_indices = tf.nn.top_k(y*y_, k=top_k)
        #self._inter_indices = tf.contrib.metrics.set_intersection(self._pred_top_indices, self._target_top_indices).values
        #self._accuracy = accuracy = inter_indices.values.eval().shape[0] / (1.0 * tf.shape(y)[0] * top_k)


      # Create saver if necessary
      if flag_with_saver:
        self.saver = tf.train.Saver(max_to_keep=None)
      else:
        self.saver = None

      # Return the model if it is just for inference
      if not is_training:
        return

      self._lr = tf.Variable(0.0, trainable=False)


    #if hasattr(config, 'optimizer'):
    #  if config.optimizer == 'ori':
    #    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    #  elif config.optimizer == 'ada': # No GPU
    #    optimizer = tf.train.AdagradOptimizer(self.lr)
    #  elif config.optimizer == 'adam':
    #    optimizer = tf.train.AdamOptimizer(self.lr)
    #  elif config.optimizer == 'rms':
    #    optimizer = tf.train.RMSPropOptimizer(self.lr)
    #  else:
    #    raise NameError("Unknown optimizer type %s!" % config.optimizer)
    #else:
    #  optimizer = tf.train.AdamOptimizer(self.lr)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self._train_op = optimizer.minimize(self._loss)


  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def aux_targets(self):
      return self._aux_targets

  @property
  def outputs(self):
    return self._outputs
    
  @property
  def logit(self):
    return self._logit

  @property
  def loss(self):
    return self._loss

  #@property
  #def accuracy(self):
  #  return self._accuracy

  @property
  def train_op(self):
    return self._train_op

  @property
  def lr(self):
    return self._lr
  
  @property
  def pred_top_values(self):
    return self._pred_top_values

  @property
  def pred_top_indices(self):
    return self._pred_top_indices

  @property
  def target_top_values(self):
    return self._target_top_values

  @property
  def target_top_indices(self):
    return self._target_top_indices

  @property
  def inter_indices(self):
    return self._inter_indices



  #@property
  #def keep_prob(self):
  #  return self._keep_prob





class MLPPredictor(object):
    """The sentence decoder (generator) for LSTMModel."""

    def __init__(self, n_layers, last_nl_fun='softmax', loss_fun='cross_entropy', top_k=10, aux_classes=0,
                 ses_threads=2, gpu_memory_fraction=0.2, gpu=1):


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        config_proto = tf.ConfigProto(intra_op_parallelism_threads=ses_threads, 
                                      gpu_options=gpu_options, 
                                      allow_soft_placement=True)
        self.session = session = tf.Session(config=config_proto)
        
        with tf.variable_scope("MLPModel"):
          self.model = MLP(is_training=False,
                           n_layers=n_layers,
                           keep_prob=0.6,
                           last_nl_fun=last_nl_fun,
                           loss_fun=loss_fun,
                           top_k=top_k,
                           aux_classes=aux_classes,
                           flag_with_saver=True,
                           gpu=gpu)

        self.model_ready = False

            
    def load_model(self, model_path):
        self.model.saver.restore(self.session, model_path)
        self.model_path = model_path
        self.model_ready = True
        

    def predict(self, input_data):
        assert(self.model_ready)
        model = self.model
        pred_top_values, pred_top_indices = self.session.run([model.pred_top_values, model.pred_top_indices],
                    feed_dict={model.input_data: input_data})

        return pred_top_values, pred_top_indices
     
