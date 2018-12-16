import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from read_data import get_loader
from data.build_vocab import Vocabulary
import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import eval_utils
import cPickle
from utils.clip_gradient import clip_gradient
from utils.adjust_learning_rate import adjust_learning_rate

import basic.path_util as path_util
import opts


import logging
logger = logging.getLogger(__file__)
formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s"
logging.basicConfig(
    format = formatter_log,
    datefmt = '%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


def main(opt, **kwargs):
    
    if opt.use_merged_vocab:
        # using merged vocabulary for Sequential Learning
        vocab_path = path_util.get_merged_vocab(opt.collection)
    else:
        vocab_path = path_util.get_vocab(opt.collection)
        
    with open(vocab_path, 'r') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    opt.vocab_size = vocab_size

    data_loader = get_loader(opt, vocab, split='train')

    model = models.setup(opt)
    model.cuda()

    epoch = 0

    checkpoint_path = path_util.get_model_dir(opt)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if opt.start_from == None:
        # if the start model is not given, 
        # check if there is a trained model in the checkpoint path
        start_info = os.path.join(checkpoint_path, 'infos.pkl')
        start_model = os.path.join(checkpoint_path, 'model.pth')
    else:
        # use the best model in the dir
        start_info = os.path.join(opt.start_from, 'infos-best.pkl')
        start_model = os.path.join(opt.start_from, 'model-best.pth')
    if os.path.exists(start_info) and os.path.exists(start_model):
        print 'RELOADING MODEL: %s'%start_model
        with open(start_info) as f:
            infos = cPickle.load(f)
            epoch = infos['epoch'] + 1
            best_val_score = infos['best_val_score']

        model.load(start_model)


    # Loss and Optimizer
    nnl_crit = nn.CrossEntropyLoss() 

    params = list(model.parameters())
    optimizer = torch.optim.Adam(
        params, lr=opt.learning_rate, weight_decay=opt.weight_decay)


    if epoch == 0:
        best_val_score = None

    current_lr = opt.learning_rate

    def train_normal():
        losses = []
        testflag = 0
        for data in tqdm(data_loader, desc='epoch %2d' % epoch):
            fc_feats = Variable(data['fc_feats'].cuda(), requires_grad=False)
            if data['att_feats'] is not None:
                att_feats = Variable(data['att_feats'].cuda(), requires_grad=False)
            else:
                att_feats = None
            lengths = data['lengths']
            captions = Variable(data['targets'].cuda(), requires_grad=False)
            mask = Variable(data['mask'].float().cuda(), requires_grad=False)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)
            targets = targets[0]

            
            # Forward, Backward and Optimize
            model.zero_grad()
            optimizer.zero_grad()

            if not opt.use_att:
                outputs, logoutputs = model(fc_feats, captions, lengths)
            else:
                outputs, logoutputs = model(fc_feats, att_feats, captions, lengths)
            outputs = pack_padded_sequence(
                outputs, lengths, batch_first=True)
            loss = nnl_crit(outputs[0], targets)

            loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            torch.cuda.synchronize()
            losses.append(loss.data[0])
            
        losses = np.array(losses)
        logger.info("train_loss = {:.3f}".format(losses.mean()))
        return losses.mean()

    # Train the Models
    while epoch < opt.num_epochs:

        # update lr
        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
            frac = (
                epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate ** frac
            current_lr = opt.learning_rate * decay_factor
        adjust_learning_rate(optimizer, current_lr)

        if opt.feedback_start_epoch >= 0 and (epoch - opt.feedback_start_epoch + 1) % opt.feedback_prob_increase_every == 0 and model.feedback_prob < opt.feedback_prob_max:
            model.feedback_prob += opt.feedback_prob_increment

        print('current_lr: ', current_lr)

        train_loss = train_normal()

        infos = {}
        # make evaluation on validation set, and save model
        # eval model
        opt.beam_size = 1
        eval_kwargs = {'split': 'val', 'id': epoch,
                       'anno_file': path_util.get_anno_file(opt.collection,split='val'),
                       'output_dir':path_util.get_output_dir(opt, 'val'),
                       'beam_size': opt.beam_size, 'verbose': False}
        val_loss, predictions, lang_stats = eval_utils.eval_split(
            opt, model, vocab, nnl_crit, eval_kwargs)

        # Save model if is improving on validation result
        current_score = lang_stats['CIDEr']
        best_flag = False
        if best_val_score is None or current_score > best_val_score:
            best_val_score = current_score
            best_flag = True

        infos['opt'] = opt
        infos['epoch'] = epoch
        infos['best_val_score'] = best_val_score

        name = '%s/model.pth' % checkpoint_path
        model.save(name=name)
        with open(os.path.join(checkpoint_path, 'infos.pkl'), 'wb') as f:
            cPickle.dump(infos, f)

        if best_flag:
            logger.info("Saving best model to %s".format(checkpoint_path))
            with open(os.path.join(checkpoint_path, 'infos-best.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            model.save(name='%s/model-best.pth' % checkpoint_path)

        epoch += 1


if __name__ == '__main__':
    opt = opts.parse_opt()
    main(opt)
