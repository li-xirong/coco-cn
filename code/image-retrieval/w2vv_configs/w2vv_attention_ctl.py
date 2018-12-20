class config(object):
    '''Configuration for w2vv model training.'''
    img_feature = 'pyresnext-101_rbps13k,flatten0_output,os'
    bow_vocab = 'word_vocab_5'
    L1_normalize = 0
    L2_normalize = 0
    sent_maxlen = 32
    lstm_size = 1024
    embed_size = 500
    we_trainable = 0
    text_style = 'gru@bow_filterstop@word2vec_filterstop'
    sequences = True

    ''' model structuer'''
    n_layers = '0-2048-2048'

    ''' loss  '''
    loss_fun = 'ctl'
    activation = 'relu'
    max_epochs = 200
    lr = 0.0001
    optimizer='rmsprop'
    l2_p = 0.000
    clipnorm = 10
    dropout = 0.2

    set_style = 'ImageSets'


