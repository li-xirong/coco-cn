import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data.build_vocab import Vocabulary
import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import eval_utils
import cPickle
from utils.clip_gradient import clip_gradient

import basic.path_util as path_util
import opts

def test(opt, **kwargs):

    if opt.use_merged_vocab:
        # using merged vocabulary for Sequential Learning
        vocab_path = path_util.get_merged_vocab(opt.collection)
    else:
        vocab_path = path_util.get_vocab(opt.collection)
        
    with open(vocab_path, 'r') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    opt.vocab_size = vocab_size

    # load model
    best_model_path = path_util.get_best_model(opt) #'%s/model-best.pth' % opt.checkpoint_path
    best_info_path = path_util.get_best_model_info(opt) #'%s/infos-best.pkl' % opt.checkpoint_path
    print(best_info_path)
    print(best_model_path)
    with open(best_info_path) as f:
        infos = cPickle.load(f)
        # opt = infos['opt']
        # best_val_score = infos['best_val_score']
    ignore = ["batch_size", "language_eval", "beam_size", "num_workers"]
    for k in vars(infos['opt']).keys():
        if k not in ignore:
            # copy over options from model
            vars(opt).update({k: vars(infos['opt'])[k]}) 

    model = models.setup(opt)
    model.cuda()

    model.load(best_model_path)

    nnl_crit = nn.CrossEntropyLoss()

    split = 'test'
    language_eval = opt.language_eval
    eval_kwargs = {'split': split, 'id': 'best-model', 'language_eval':language_eval,
                   'anno_file': path_util.get_test_anno_file(opt.collection),
                   'output_dir':path_util.get_output_dir(opt, split),
                   'beam_size': opt.beam_size, 
                   'verbose': True}
    test_loss, predictions, lang_stats = eval_utils.eval_split(
        opt, model, vocab, nnl_crit, eval_kwargs)


if __name__ == '__main__':
    #opt = DefaultConfig()
    opt = opts.parse_opt()
    test(opt)
