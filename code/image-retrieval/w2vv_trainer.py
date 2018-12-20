from __future__ import print_function

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

import os
import sys
import time
import random
import numpy as np

from keras.utils import generic_utils
from keras.preprocessing.sequence import pad_sequences

from basic.constant import *
from basic.metric import getScorer
from basic.common import makedirsforfile, checkToSkip

from w2vv import W2VV_MS

from simpleknn.bigfile import BigFile
from utils.dataset import get_dataset
from utils.simer import get_simer
from utils.text2vec import get_text_encoder
from utils.text import get_we_parameter
from utils.util import check_img_list, readSentsInfo, load_config, which_language

import tensorflow as tf                                 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

INFO = __file__

def do_validation(val_img_list, val_img_feats, val_sents_id, sent_feats_1, sent_feats_2, simer, scorer, model):
    logger.info('Validating...')
    perf_list = []
    val_progbar = generic_utils.Progbar(len(val_img_list))
    projected_sents = model.predict_batch(sent_feats_1,  sent_feats_2)

    val_batch_size = 1000

    for img_start in range(0, len(val_img_list), val_batch_size):
        img_end = min(len(val_img_list), img_start+val_batch_size)
        img_batch_list = val_img_list[img_start: img_end]

        renamed_batch, img_feat_batch = val_img_feats.read(img_batch_list)
        scorelist = simer.calculate(img_feat_batch, projected_sents)
        
        for idx, current_score_list in enumerate(scorelist):
            sorted_sent_score = sorted(zip(val_sents_id, current_score_list), key = lambda a: a[1], reverse=True)
            sorted_sent = [x[0] for x in sorted_sent_score]
            sorted_label = map(check_img_list, sorted_sent, [renamed_batch[idx]]*len(sorted_sent))

            # Average precision
            current_score = scorer.score(sorted_label)
            # Inverted rank
            # current_score = 1.0 / (sorted_label.index(1)+1)
            perf_list.append(current_score)
            val_progbar.update(len(perf_list), values=[("perf %s" % scorer.name(), current_score)])

    assert len(perf_list) == len(val_img_list)
    return  np.mean(perf_list)



def process(options, trainCollection, valCollection, testCollection):
    lang = which_language(trainCollection)
    assert(which_language(trainCollection) == which_language(valCollection))
    assert(which_language(trainCollection) == which_language(testCollection))

    rootpath = options.rootpath
    overwrite =  options.overwrite
    checkpoint = options.checkpoint
    init_model_from = options.init_model_from
    unroll = options.unroll
    corpus = options.corpus
    word2vec = options.word2vec
    batch_size = options.batch_size
    
    w2vv_config = options.model_config
    config = load_config('w2vv_configs/%s.py' % w2vv_config)

    img_feature = config.img_feature
    set_style = config.set_style
    # text embedding style (word2vec, bag-of-words, word hashing)
    text_style = config.text_style
    L1_normalize = config.L1_normalize
    L2_normalize = config.L2_normalize
    
    bow_vocab = config.bow_vocab+'.txt'

    l2_p = config.l2_p
    dropout = config.dropout
    
    max_epochs= config.max_epochs
    optimizer = config.optimizer
    loss_fun = config.loss_fun
    lr = config.lr
    clipnorm = config.clipnorm
    activation = config.activation
    sequences = config.sequences

    # lstm
    sent_maxlen = config.sent_maxlen
    embed_size = config.embed_size
    we_trainable = config.we_trainable
    lstm_size = config.lstm_size

    n_layers = map(int, config.n_layers.strip().split('-'))

    if init_model_from != '':
        init_model_name = init_model_from.strip().split("/")[-1]
        train_style = INFO + "_" + init_model_name
    else:
        train_style = INFO

    rnn_style, bow_style, w2v_style = text_style.strip().split('@')
    
    # text embedding style
    model_info = w2vv_config

    if 'lstm' in text_style or 'gru' in text_style:
        if lang == 'zh':
            w2v_data_path = os.path.join(rootpath, 'zh_w2v', 'model', 'zh_jieba.model')
        else:
            w2v_data_path = os.path.join(rootpath, "word2vec", corpus, word2vec)

        # bag-of-words vocabulary file path
        text_data_path = os.path.join(rootpath, trainCollection, "TextData", "vocabulary", "bow", bow_vocab)
        bow_data_path = os.path.join(rootpath, trainCollection, "TextData", "vocabulary", bow_style, bow_vocab)
        
        # text embedding (text representation)
        text2vec = get_text_encoder(rnn_style)(text_data_path, ndims=0, language=lang, L1_normalize=L1_normalize, L2_normalize=L2_normalize, maxlen=sent_maxlen)
        bow2vec = get_text_encoder(bow_style)(bow_data_path, ndims=0, language=lang, L1_normalize=L1_normalize, L2_normalize=L2_normalize)
        w2v2vec = get_text_encoder(w2v_style)(w2v_data_path, ndims=0, language=lang, L1_normalize=L1_normalize, L2_normalize=L2_normalize)
        if n_layers[0] == 0:
            n_layers[0] = bow2vec.ndims + w2v2vec.ndims
        else:
            assert n_layers[0] == bow2vec.ndims + w2v2vec.ndims

        # log file
        checkpoint_dir = os.path.join(rootpath, trainCollection, checkpoint, valCollection, train_style, model_info)

    else:
        logger.info("%s is not supported, please check the 'text_style' parameter", text_style)
        sys.exit(0)

    train_loss_hist_file = os.path.join(checkpoint_dir, 'train_loss_hist.txt')
    val_per_hist_file = os.path.join(checkpoint_dir, 'val_per_hist.txt')
    model_file_name = os.path.join(checkpoint_dir, 'model.json')
    model_img_name = os.path.join(checkpoint_dir, 'model.png')

    logger.info(model_file_name)
    if checkToSkip(model_file_name, overwrite):
        sys.exit(0)

    makedirsforfile(val_per_hist_file)

    # img2vec
    img_feat_path = os.path.join(rootpath, FULL_COLLECTION, 'FeatureData', img_feature)
    img_feats = BigFile(img_feat_path)

    val_img_feat_path = os.path.join(rootpath, FULL_COLLECTION, 'FeatureData', img_feature)
    val_img_feats = BigFile(val_img_feat_path)

    # dataset 
    train_file = os.path.join(rootpath, trainCollection, 'TextData', '%s.caption.txt' % trainCollection)

    # training set
    # print "loss function: ", loss_fun
    dataset_style = 'sent_' + loss_fun
    DataSet = get_dataset(dataset_style)

    # represent text on the fly
    trainData = DataSet(train_file, batch_size, text2vec, bow2vec, w2v2vec, img_feats, flag_maxlen=True, maxlen=sent_maxlen)

    # get pre-trained word embedding
    we_weights = get_we_parameter(text2vec.vocab, w2v_data_path, lang)
    # define word2visualvec model
    w2vv = W2VV_MS( text2vec.nvocab, sent_maxlen, embed_size, we_weights, we_trainable, lstm_size, n_layers, dropout, l2_p, activation=activation, lstm_style=rnn_style, sequences=sequences, unroll=unroll)

    w2vv.save_json_model(model_file_name)
    w2vv.plot(model_img_name)
    w2vv.compile_model(optimizer, loss_fun, learning_rate = lr, clipnorm=clipnorm)
   

    if options.init_model_from != '':
        logger.info('initialize the model from %s', options.init_model_from)
        w2vv.init_model(options.init_model_from)

    # preparation for validation
    val_sent_file = os.path.join(rootpath, valCollection, 'TextData', '%s.caption.txt' % valCollection)
    val_sents_id, val_sents, val_id2sents = readSentsInfo(val_sent_file)
    val_img_list = map(str.strip, open(os.path.join(rootpath, valCollection,  set_style, '%s.txt' % valCollection)).readlines())

    sent_feats_1 = []
    sent_feats_2 = []
    new_val_sents_id = []
    for index, sent in enumerate(val_sents):
        sent_vec = text2vec.mapping(sent)
        bow_vec = bow2vec.mapping(sent)
        w2v_vec = w2v2vec.mapping(sent)
        if sent_vec is not None and bow_vec is not None and w2v_vec is not None:
            sent_feats_1.append(sent_vec)
            sent_feats_2.append(list(bow_vec) + list(w2v_vec))
            new_val_sents_id.append(val_sents_id[index])
    sent_feats_1 = pad_sequences(sent_feats_1, maxlen=sent_maxlen,  truncating='post')

    simer = get_simer('cosine_batch')()
    scorer = getScorer(options.val_metric)

    count = 0
    lr_count = 0
    best_validation_perf = 0
    best_epoch = -1
    train_loss_hist = []
    val_per_hist = []
    n_train_batches = int(np.ceil( 1.0 * trainData.datasize / batch_size ))
    if loss_fun == 'ctl':
        datasize = 2*trainData.datasize
    else:
        datasize = trainData.datasize
    for epoch in range(max_epochs):
        logger.info('Epoch %d', epoch)
        logger.info("Training..., learning rate: %g", w2vv.get_lr())
        
        train_loss_epoch = []
        train_progbar = generic_utils.Progbar(datasize)
        trainBatchIter = trainData.getBatchData()
        for minibatch_index in xrange(n_train_batches):
            train_X_batch, train_Y_batch = trainBatchIter.next()
            loss = w2vv.model.train_on_batch(train_X_batch, train_Y_batch)
            train_progbar.add(train_X_batch[0].shape[0], values=[("train loss", loss)])

            train_loss_epoch.append(loss)

        train_loss_hist.append(np.mean(train_loss_epoch))

        this_validation_perf = do_validation(val_img_list, val_img_feats, new_val_sents_id, sent_feats_1, sent_feats_2, simer, scorer, w2vv)
        val_per_hist.append(this_validation_perf)

        logger.info('previous_best_performance: %g', best_validation_perf)
        logger.info('current_performance: %g', this_validation_perf)

        fout_file = os.path.join(checkpoint_dir, 'epoch_%d.h5' % ( epoch))

        lr_count += 1
        if this_validation_perf > best_validation_perf:
            best_validation_perf = this_validation_perf          
            count = 0

            # save best model
            w2vv.model.save_weights(fout_file)
            if best_epoch != -1:
                os.system('rm '+ os.path.join(checkpoint_dir, 'epoch_%d.h5' % (best_epoch)))
            best_epoch = epoch
        else:
            # when the validation performance has decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs.
            if lr_count > 2:
                w2vv.decay_lr(0.5)
                lr_count = 0
            count += 1
            if count > 10:
                print ("Early stopping happend")
                break


    sorted_epoch_loss = zip(range(len(train_loss_hist)), train_loss_hist)
    with open(train_loss_hist_file, 'w') as fout:
        for i, loss in sorted_epoch_loss:
            fout.write("epoch_" + str(i) + " " + str(loss) + "\n")


    sorted_epoch_perf = sorted(zip(range(len(val_per_hist)), val_per_hist), key = lambda x: x[1], reverse=True)
    with open(val_per_hist_file, 'w') as fout:
        for i, perf in sorted_epoch_perf:
            fout.write("epoch_" + str(i) + " " + str(perf) + "\n")


    # generate the shell script for test
    templete = ''.join(open( 'TEMPLATE_do_test.sh').readlines())
    striptStr = templete.replace('@@@rootpath@@@', rootpath)
    striptStr = striptStr.replace('@@@overwrite@@@', str(overwrite))
    striptStr = striptStr.replace('@@@trainCollection@@@', trainCollection)
    striptStr = striptStr.replace('@@@testCollection@@@', '%s %s'%(valCollection, testCollection))
    striptStr = striptStr.replace('@@@model_config@@@', w2vv_config)
    striptStr = striptStr.replace('@@@set_style@@@', set_style)
    striptStr = striptStr.replace('@@@model_path@@@', checkpoint_dir)
    striptStr = striptStr.replace('@@@model_name@@@', 'model.json')
    striptStr = striptStr.replace('@@@weight_name@@@', 'epoch_%d.h5' % sorted_epoch_perf[0][0])
    runfile = 'do_test_%s_%s.sh' % (w2vv_config, testCollection)
    open( runfile, 'w' ).write(striptStr+'\n')
    os.system('chmod +x %s' % runfile)
    os.system('./%s' % runfile)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection valCollection testCollection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)

    parser.add_option('--checkpoint', type="string", default='cv_keras', help='output directory to write checkpoints to (default: cv_keras)')
    parser.add_option('--init_model_from', type="string", default='', help='initialize the model parameters from some specific checkpoint?')

    # word2vector parameters
    parser.add_option("--corpus", default=DEFAULT_CORPUS, type="string", help="corpus using which word2vec was trained (default: %s)" % DEFAULT_CORPUS)
    parser.add_option("--word2vec", default=DEFAULT_WORD2VEC, type="string", help="word2vec model (default: %s)" % DEFAULT_WORD2VEC)

    # model parameters
    parser.add_option('--unroll', type="int", default=1, help='size of the lstm (default: 512)')

    # optimization parameters
    parser.add_option('--batch_size', type="int", default=100, help='batch size (default: 100)')

    parser.add_option('--val_metric', type="str", default='AP', help='performance metric (default: AP)')
    parser.add_option('--model_config', type="str", default='base_w2vv', help='model configuration file (default: base_w2vv)')
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    return process(options, args[0], args[1], args[2])


if __name__ == "__main__":
    sys.exit(main())
