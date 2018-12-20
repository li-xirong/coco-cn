from __future__ import print_function

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

import os
import re
import sys
import numpy as np

from gensim.models import word2vec as w2v

from basic.constant import *
from basic.common import makedirsforfile, checkToSkip

from keras.utils import generic_utils
from keras.preprocessing.sequence import pad_sequences

from w2vv_pred import W2VV_pred

from simpleknn.bigfile import BigFile
from utils.simer import get_simer
from utils.text2vec  import get_text_encoder
from utils.eval_perf import cal_perf_t2i
from utils.util import check_img_list, readSentsInfo, load_config, which_language

import tensorflow as tf                                 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


def process(options, trainCollection, testCollection):
    lang = which_language(trainCollection)
    assert(which_language(trainCollection) == which_language(testCollection))

    rootpath = options.rootpath
    overwrite = options.overwrite
    
    model_path = options.model_path
    model_name = options.model_name
    weight_name = options.weight_name
    resfile = options.resfile

    # only save the predicted top k sentence
    k = options.k

    corpus = options.corpus
    word2vec = options.word2vec
    simi_fun = options.simi_fun
    set_style = options.set_style

    w2vv_config = os.path.basename(os.path.normpath(model_path))
    config = load_config('w2vv_configs/%s.py' % w2vv_config)

    # image feature
    img_feature = config.img_feature
    
    # text embedding style (word2vec, bag-of-words, word hashing)
    text_style = config.text_style
    L1_normalize = config.L1_normalize
    L2_normalize = config.L2_normalize

    bow_vocab = config.bow_vocab+'.txt'

    loss_fun = config.loss_fun


    # lstm
    sent_maxlen = config.sent_maxlen
    embed_size = config.embed_size
    we_trainable = config.we_trainable

    # result file info
    output_dir = os.path.join(rootpath, testCollection, 'SimilarityIndex', trainCollection, w2vv_config)
    result_pred_sents = os.path.join(output_dir, 'sent.id.score.txt')
    sent_feat_file = os.path.join(output_dir, "sent_feat.txt")

    test_sent_file = os.path.join(rootpath, testCollection, 'TextData', '%s.caption.txt'%testCollection)

    if checkToSkip(sent_feat_file, overwrite):
        sys.exit(0)
    makedirsforfile(result_pred_sents)

    rnn_style, bow_style, w2v_style = text_style.strip().split('@')

    if "lstm" in text_style or "gru" in text_style:
        if 'zh' == lang:
            w2v_data_path = os.path.join(rootpath, 'zh_w2v', 'model', 'zh_jieba.model')
        else:
            w2v_data_path = os.path.join(rootpath, "word2vec", corpus, word2vec)
        text_data_path = os.path.join(rootpath, trainCollection, "TextData", "vocabulary", 'bow', bow_vocab)
        bow_data_path = os.path.join(rootpath, trainCollection, "TextData", "vocabulary", bow_style, bow_vocab)

        text2vec = get_text_encoder(rnn_style)(text_data_path, ndims=0, language=lang, L1_normalize=L1_normalize, L2_normalize=L2_normalize, maxlen=sent_maxlen)
        bow2vec = get_text_encoder(bow_style)(bow_data_path, ndims=0, language=lang, L1_normalize=L1_normalize, L2_normalize=L2_normalize)
        w2v2vec = get_text_encoder(w2v_style)(w2v_data_path, ndims=0, language=lang, L1_normalize=L1_normalize,L2_normalize=L1_normalize)
    else:
        logger.info("%s is not supported, please check the 'text_style' parameter", text_style)
        sys.exit(0)

    # img2vec
    img_feats_path = os.path.join(rootpath, FULL_COLLECTION, 'FeatureData', img_feature)
    img_feats = BigFile(img_feats_path)

    # similarity function
    simer = get_simer(simi_fun)()

    abs_model_path = os.path.join(model_path, model_name)
    weight_path = os.path.join(model_path, weight_name)
    predictor = W2VV_pred(abs_model_path, weight_path, text2vec, sent_maxlen, embed_size, bow2vec, w2v2vec)

    test_sents_id, test_sents, id2sents = readSentsInfo(test_sent_file)
    test_img_list = map(str.strip, open(os.path.join(rootpath, testCollection, set_style, '%s.txt' % testCollection)).readlines())
    
    fw = open(sent_feat_file, 'w')
    logger.info("predict the visual CNN features for all sentences in the test set ...")
    pred_progbar = generic_utils.Progbar(len(test_sents_id))
    filtered_test_sent_id = []
    test_sent_visual_feats_batch_list = []
    text_batch_size = 10000
    for start in range(0, len(test_sents_id), text_batch_size):
        end = min(len(test_sents_id), start+text_batch_size)
        text_batch_list = test_sents_id[start: end]

        test_sent_visual_feats_batch = []
        sents_id = []
        for index in range(len(text_batch_list)):
            sid = text_batch_list[index]
            test_sent = id2sents[sid]
            test_sent_feat = predictor.predict_one(test_sent)
            if test_sent_feat is not None:
                test_sent_visual_feats_batch.append(test_sent_feat)
                sents_id.append(sid)
                fw.write(sid + ' ' + ' '.join(map(str, test_sent_feat)) + '\n')
            else:
                logger.info('failed to vectorize "%s"', test_sent)
            pred_progbar.add(1)
        test_sent_visual_feats_batch_list.append(test_sent_visual_feats_batch)
        filtered_test_sent_id.append(sents_id)

    fw.close()
 
    # evaluation only when training on Chinese collection
    if 'zh' == lang:
        logger.info("matching image and text on %s ...", testCollection)
        fout_1 = open(result_pred_sents, 'w')
        test_progbar = generic_utils.Progbar(len(test_img_list))

        img_batch_size = 1000
        counter = 0
        for i, test_sent_visual_feats in enumerate(test_sent_visual_feats_batch_list):
            
            sents_id = filtered_test_sent_id[i]
            batch_score_list = []
            for start in range(0, len(test_img_list), img_batch_size):
                end = min(len(test_img_list), start+img_batch_size)
                img_batch_list = test_img_list[start: end]

                img_feat_batch = []
                for test_img in img_batch_list:
                    test_img_feat = img_feats.read_one(test_img)
                    img_feat_batch.append(test_img_feat)

                scorelist_batch = simer.calculate(test_sent_visual_feats, img_feat_batch)
                batch_score_list.append(scorelist_batch)
            batch_score_list = np.concatenate(batch_score_list, axis = -1)

            assert len(batch_score_list) == len(sents_id)

            for sent_id, scorelist in zip(sents_id, batch_score_list):
                top_hits = np.argsort(scorelist)[::-1]

                top_imgs = []
                for idx in top_hits.tolist():
                    top_imgs.append(test_img_list[idx]) 
                    top_imgs.append(scorelist[idx])

                fout_1.write(sent_id + ' ' + ' '.join(map(str,top_imgs)) + '\n')

                counter += 1
                test_progbar.update(counter)
        assert counter == len(test_sents_id)

        fout_1.close()
        print (result_pred_sents)
        recall_name, recall_score, med_r, mean_r, mean_invert_r = cal_perf_t2i(result_pred_sents)
        #fout_recall = open(os.path.join(output_dir, 'recall.txt'), 'w')
        #fout_recall.write(recall_name + '\n' + recall_score + '\n')
        #fout_recall.write('med_r:' + '\n' + str(med_r) + '\n')
        #fout_recall.write('mean_r:' + '\n' + str(mean_r) + '\n')
        #fout_recall.write('mean_invert_r:' + '\n' + str(mean_invert_r) + '\n')
        fout_recall = open(os.path.join(output_dir, 'mir.txt'), 'w')
        fout_recall.write('mean_invert_r: {}\n'.format(round(mean_invert_r, 3))) 
        fout_recall.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection testCollection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    
    # word2vector parameters
    parser.add_option("--corpus", default=DEFAULT_CORPUS, type="string", help="corpus using which word2vec is trained (default: %s)" % DEFAULT_CORPUS)
    parser.add_option("--word2vec", default=DEFAULT_WORD2VEC, type="string", help="word2vec model (default: %s)" % DEFAULT_WORD2VEC)
    # bag-of-words parameters
    parser.add_option("--bow_vocab", default=DEFAULT_BOW_VOCAB, type="string", help="bag-of-words vocabulary file name (default: %s)"%DEFAULT_BOW_VOCAB)
    # trained models
    parser.add_option("--model_path", default='rootpath/trainCollection/cv_keras/valCollection/train_style', type="string", help="model path")
    parser.add_option("--model_name", default='model.josn', type="string", help="model name")
    parser.add_option("--weight_name", default='epoch_100.h5', type="string", help="weight name")

    # similarity function
    parser.add_option("--simi_fun", default='cosine', type="string", help="similarity function: dot cosine L1 L2")

    parser.add_option("--k", default="100", type="int", help="top k predicted sentence")
    parser.add_option("--set_style", default="ImageSets", type="string", help="training data filename (default: train_pair.txt)")
    parser.add_option("--resfile", default="set.2.A.txt", type="string", help="trecvid submit file (default: set.2.A.txt)")
    

    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())    
