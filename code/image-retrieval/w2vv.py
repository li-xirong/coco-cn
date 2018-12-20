import numpy as np 
np.random.seed(1337)

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, RepeatVector, Permute
from keras.layers import BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam
from keras.regularizers import l2

from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import LSTM
if  keras.__version__.startswith('1'):
    from keras.utils.visualize_util import plot
else:
    from keras.utils import plot_model as plot

from keras.layers import Input, Bidirectional, GRU, TimeDistributed, Activation, merge, concatenate, multiply
from keras.models import Model

import numpy as np
import tensorflow as tf


def l2norm(X):
    """L2 norm, row-wise"""
    norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
    X /= norm[:, None]
    return X

# cosine based margin ranking loss
def cosine_mrl_option(labels, predicts):
    """For a minibatch of image and sentences embeddings, computes the pairwise contrastive loss"""
    #batch_size, double_n_emd = tensor.shape(predicts)
    #res = tensor.split(predicts, [double_n_emd/2, double_n_emd/2], 2, axis=-1)

    img = l2norm(labels)
    text = l2norm(predicts)
    scores = tensor.dot(img, text.T)

    diagonal = scores.diagonal()
    mrl_margin = 0.3
    loss_max_violation = True

    # caption retrieval (1 + neg - pos)
    cost_s = tensor.maximum(0, mrl_margin + scores - diagonal.reshape((-1,1)))
    # clear diagonals
    cost_s = fill_diagonal(cost_s, 0)

    # img retrieval
    cost_im = tensor.maximum(0, mrl_margin + scores - diagonal)
    cost_im = fill_diagonal(cost_im, 0)

    if loss_max_violation:
        if cost_s:
            cost_s = tensor.max(cost_s, axis=1)
        if cost_im:
            cost_im = tensor.max(cost_im, axis=0)

    loss = cost_s.mean() + cost_im.mean()
    return loss

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def mean_squared_error(labels, predicts):
    return K.mean(K.square(labels-predicts), axis=-1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 2
    batch_size_shape = tf.shape(y_true)
    pair_flag, video_feat = tf.split(y_true, [1, batch_size_shape[1]-1], axis=-1)
    distance = euclidean_distance([y_pred, video_feat])
    return K.mean(pair_flag * K.square(distance) +
                  (1 - pair_flag) * K.square(K.maximum(margin - distance, 0)))
    '''
    batch_size, double_n_emd = tensor.shape(y_true)
    pair_flag, video_feat = tensor.split(y_true, [1, double_n_emd-1], 2, axis=-1)
    distance = euclidean_distance([video_feat, y_pred])
    return K.mean(pair_flag * K.square(distance) +
                  (1 - pair_flag) * K.square(K.maximum(margin - distance, 0)))
    '''

# basic word2visualvec
class W2VV:
    def compile_model(self, optimizer='rmsprop', loss='mse', learning_rate=0.001, clipnorm=0):
        print "loss function: ", loss
        print "learning_rate: ", learning_rate
        print "clipnorm:", clipnorm

        if loss == 'mrl':
            loss = cosine_mrl_option
        elif loss == 'ctl':
            loss = contrastive_loss

        if optimizer == 'sgd':
            # let's train the model using SGD + momentum (how original).
            if clipnorm > 0:
                sgd = SGD(lr=learning_rate, clipnorm=clipnorm, decay=1e-6, momentum=0.9, nesterov=True)
            else:
                sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss=loss, optimizer=sgd)
        elif optimizer == 'rmsprop':
            if clipnorm > 0:
                rmsprop = RMSprop(lr=learning_rate, clipnorm=clipnorm, rho=0.9, epsilon=1e-6)
            else:
                rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
            self.model.compile(loss=loss, optimizer=rmsprop)
        elif optimizer == 'adagrad':
            if clipnorm > 0:
                adagrad = Adagrad(lr=learning_rate, clipnorm=clipnorm, epsilon=1e-06)
            else:
                adagrad = Adagrad(lr=learning_rate, epsilon=1e-06)
            self.model.compile(loss=loss, optimizer=adagrad)
        elif optimizer == 'adam':
            if clipnorm > 0:
                adam = Adam(lr=learning_rate, clipnorm=clipnorm, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            else:
                adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            self.model.compile(loss=loss, optimizer=adam)

            
    def init_model(self, fname):
        self.model.load_weights(fname)

    def predict_one(self, X1, X2):
        X1 = np.array([X1]).astype('float32')
        X2 = np.array([X2]).astype('float32')
        return self.model.predict([X1, X2], verbose=0)[0]

    def predict_batch(self, X1, X2):
        X1 = np.array(X1).astype('float32')
        X2 = np.array(X2).astype('float32')
        return self.model.predict([X1, X2], verbose=0)

    def save_json_model(self, model_file_name):
        json_string = self.model.to_json()
        if model_file_name[-5:] != '.json':
            model_file_name = model_file_name + '.json'
        open(model_file_name, 'w').write(json_string)

    def plot(self, filename):
        plot(self.model, to_file=filename, show_shapes=True, show_layer_names=True)

    def get_lr(self):
        return K.eval(self.model.optimizer.lr)

    def decay_lr(self, decay=0.9):
        old_lr = self.get_lr()
        new_lr = old_lr * decay
        K.set_value(self.model.optimizer.lr, new_lr)



# word2visualvec that has input, output layer and at least one hidden layer
class W2VV_MS(W2VV):
    def __init__(   
        self, 
        vocab_size,
        sent_maxlen,
        embed_size=512,
        we_weights=None,
        we_trainable=0,
        lstm_size=512,
        n_layers=[512,1000,2000,4096],
        dropout=0.2, 
        l2_p = 0.0,
        have_relu_last=1,
        activation='relu',
        lstm_style = 'lstm',
        sequences = False,
        unroll=1
        ):

        we_trainable = False if we_trainable == 0 else True
        unroll = False if unroll == 0 else True

        print 'vocab_size', vocab_size
        print 'sent_maxlen', sent_maxlen
        print 'embed_size', embed_size
        print 'we_trainable', we_trainable
        print 'n_layers:', n_layers
        print 'dropout:', dropout
        print 'l2_p:', l2_p
        print 'activation:', activation
        print 'lstm_style:', lstm_style


        assert (lstm_style in ['lstm', 'bilstm', 'gru', 'bigru']), "not supported LSTM style (%s)" % lstm_style
        assert (activation in ['relu', 'prelu']), "not supported activation (%s)" % activation

        # creat model
        print("Building model...")

        main_input = Input(shape=(sent_maxlen,))

        if we_weights is None:
            we = Embedding(vocab_size, embed_size)(main_input)
        else:
            we = Embedding(vocab_size, embed_size, trainable=we_trainable, weights=[we_weights])(main_input)
        we_dropout = Dropout(dropout)(we)

        if lstm_style == 'lstm':
            lstm_out = LSTM(lstm_size, return_sequences=sequences, unroll=unroll, consume_less='gpu', init='glorot_uniform')(we_dropout)
        elif lstm_style == 'bilstm':
            lstm_out = Bidirectional(LSTM(lstm_size, return_sequences=sequences, unroll=unroll, consume_less='gpu', init='glorot_uniform'))(we_dropout)
        elif lstm_style == 'gru':
            lstm_out = GRU(lstm_size, return_sequences=sequences, unroll=unroll, implementation=2, kernel_initializer='glorot_uniform')(we_dropout)
        elif lstm_style == 'bigru':
            lstm_out = Bidirectional(GRU(lstm_size, return_sequences=sequences, unroll=unroll, consume_less='gpu', init='glorot_uniform'))(we_dropout)
        lstm_out_dropout = Dropout(dropout)(lstm_out)

        if sequences:
            lstm_out_dropout = apply_attention(lstm_out_dropout, lstm_size)

        # bow, word2vec embedded sentence vector
        auxiliary_input = Input(shape=(n_layers[0],))

        x = concatenate([lstm_out_dropout, auxiliary_input], axis=-1)
        for n_neuron in range(1,len(n_layers)-1):
            x = Dense(n_layers[n_neuron], kernel_regularizer=l2(l2_p), kernel_initializer='glorot_uniform')(x)
            x = Activation(activation)(x)
            x = Dropout(dropout)(x)
        
        x = Dense(n_layers[-1], kernel_regularizer=l2(l2_p), kernel_initializer='glorot_uniform')(x)
        output = Activation(activation)(x)

        self.model = Model([main_input, auxiliary_input], output)
        self.model.summary()


def apply_meanpool(rnn_input, outshape=None):
    out = Lambda(lambda xin: K.mean(xin, axis=1), output_shape=outshape)(rnn_input)
    return out

def apply_attention(rnn_out, lstm_size):
    attention = TimeDistributed(Dense(1, activation='tanh'))(rnn_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(lstm_size)(attention)
    attention = Permute([2,1])(attention)

    elem_wise = multiply([attention, rnn_out])
    out = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(lstm_size, ))(elem_wise)

    return out

if __name__ == '__main__':
    pair_flag = np.ones((10,1))
    video_feat = np.random.random((10,500))
    y_pred = np.random.random((10,500))
    y_true = [pair_flag, video_feat]

    ret = contrastive_loss(y_true, y_pred)
    print ret
