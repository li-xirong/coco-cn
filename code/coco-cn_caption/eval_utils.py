import torch
import numpy as np
import os
import pickle
from read_data import get_loader
from data.build_vocab import Vocabulary
import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import json
from json import encoder
from utils.decode_seq import decode_sequence
import codecs

import sys
reload(sys)
sys.setdefaultencoding('utf8')

# using caption evaluation scripts from AIChallenge
# because it's indexed by filename, not image id
from evals.coco_caption.pycxtools.coco import COCO
from evals.coco_caption.pycxevalcap.eval import COCOEvalCap

import logging
logger = logging.getLogger(__file__)
formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s"
logging.basicConfig(
    format = formatter_log,
    datefmt = '%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


#def language_eval(opt, preds, model_id, split):
def language_eval(cache_path, anno_file, output_file):
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    coco = COCO(anno_file)
    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for x in imgToEval:
        imgToEval[x]['caption'] = cocoRes.imgToAnns[x]

    with open(output_file, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    logger.info("imgToEval saved to {}".format(output_file))

    return out


def eval_split(opt, model, vocab, crit, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    split = eval_kwargs.get('split')
    lang_eval = eval_kwargs.get('language_eval', 1)
    if lang_eval:
        anno_file = eval_kwargs['anno_file']
    beam_size = eval_kwargs.get('beam_size', 1)
    print('beam_size:', beam_size)

    output_dir = eval_kwargs['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if beam_size>1 and split == 'test':
        fw = codecs.open(os.path.join(output_dir, 'beam_output.txt'),'w','utf-8')

    data_loader = get_loader(opt, vocab, split=split)
    # Make sure in the evaluation mode
    model.eval()

    def eval_nomal():
        loss_sum = 0
        loss_evals = 1e-8
        predictions = []
        test_num = 0
        evaluate_imgs = set()
        #for (features, captions, lengths, masks, cocoids, _) in tqdm(data_loader):
        for data in tqdm(data_loader):
            
            img_ids = data['img_ids']
            img_names = data['img_names']

            # forward the model to get loss
            lengths = data['lengths']
            vfc_feats = Variable(data['fc_feats'].cuda())
            if data['att_feats'] is not None:
                vatt_feats = Variable(data['att_feats'].cuda(), requires_grad=False)
            else:
                vatt_feats = None
            vcaptions = Variable(data['targets'].cuda())
            targets = pack_padded_sequence(
                vcaptions, lengths, batch_first=True)
            targets = targets[0]

            if not opt.use_att:
                outputs, logoutputs = model(vfc_feats, vcaptions, lengths)
            else:
                outputs, logoutputs = model(vfc_feats, vatt_feats, vcaptions, lengths)
            # get loss
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True)
            #loss = crit(outputs[0], targets).data[0]
            loss = crit(outputs[0], targets).item()
            # loss = crit(outputs, targets).item()
            loss_sum = loss_sum + loss
            # the num of loss
            loss_evals = loss_evals + 1


            # forward the model to also get generated samples for each image

            # we load data as image-sentence pairs
            # but we only need to forward each image once for evaluation
            # so we get the image set and mask out same images with feature_mask
            feature_mask = []
            for img_id in img_ids:
                feature_mask.append(img_id not in evaluate_imgs)
                evaluate_imgs.add(img_id)

            img_ids = np.array(img_ids)[feature_mask]
            img_names = np.array(img_names)[feature_mask]
            fc_feats = (data['fc_feats'].numpy())[feature_mask]
            if fc_feats.shape[0] == 0:
                continue
            fc_feats = torch.from_numpy(fc_feats)
            fc_feats = Variable(fc_feats).cuda()
            if opt.use_att:
                att_feats = (data['att_feats'].numpy())[feature_mask]
                att_feats = torch.from_numpy(att_feats)
                att_feats = Variable(att_feats).cuda()

            if beam_size > 1 and split == 'test':
                if opt.use_att:
                    beam_seq, beam_logprob = model.sample_beam(fc_feats, att_feats, eval_kwargs)
                else:
                    beam_seq, beam_logprob = model.sample_beam(fc_feats, eval_kwargs)
                seq = beam_seq[:,0,:] # Get the first beam for each image
                for k, beams in enumerate(beam_seq):
                    sents = decode_sequence(vocab, beams)
                    img_name = img_names[k].split('.')[0]
                    fw.write('%s\t'%img_name)
                    for i in range(beam_size):
                        fw.write('%.3f\t%s\t'%(sum(beam_logprob[k][i]) ,sents[i]))
                    fw.write('\n')
            else:
                if opt.use_att:
                    seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)
                else:
                    seq, _ = model.sample(fc_feats, eval_kwargs)

            # set_trace()
            sents = decode_sequence(vocab, seq)

            for index, sent in enumerate(sents):
                res = {'image_id': img_ids[index], 
                       'file_name': img_names[index]+'.jpg','caption': sent}
                predictions.append(res)
                if verbose:
                    print('image %s: %s' % (res['image_id'], res['caption']))

            if verbose:
                print('evaluating validation performance...  (%f)' % (loss))

        return loss_sum, loss_evals, predictions

    loss_sum, loss_evals, predictions = eval_nomal()

    # serialize to temporary json file.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    cache_path = os.path.join(output_dir, \
                              'auto_caption_%s.json'%str(eval_kwargs['id']))
    json.dump(predictions, open(cache_path, 'w'))

    lang_stats = None
    if lang_eval == 1:
        eval_result_path = os.path.join(output_dir, \
                               'auto_eval_%s.json'%str(eval_kwargs['id']))
        lang_stats = language_eval(cache_path, anno_file, eval_result_path)


    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats
