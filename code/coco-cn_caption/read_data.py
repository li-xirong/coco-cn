import torch
import torch.utils.data as data
import os
import numpy as np
import json
import random
from torch.utils.data.sampler import RandomSampler

import basic.path_util as path_util 
from basic.common import ROOT_PATH as rootpath 
from simpleknn.bigfile import BigFile


class CocoDataset(data.Dataset):

    def __init__(self, input_json_path, vocab, vf_dir, use_att=False, eng_gt_file=None, rootpath=rootpath):

        print input_json_path
        with open(input_json_path) as f:
            data = json.load(f)

        self.eng_gt_file = eng_gt_file
        self.imgname2enggt = {}
        if self.eng_gt_file is not None:
            assert os.path.exists(self.eng_gt_file), "Eng gt file not exist: %s"%eng_gt_file    
            print ('Loading eng gt file')
            eng_data = json.load(open(self.eng_gt_file))
            for x in eng_data['images']:
                img_filename = x['filename']
                sents=[]
                for y in x['sentences']:
                    sents.append(' '.join(y['tokens']))
                self.imgname2enggt[img_filename] = sents

        self.images = data['images']
        self.vocab = vocab
        self.sentences = {}
        self.img2sents = {}
        self.img2enggt = {}
        self.img2filename = {}
        self.sentId2imgId = {}
        self.imgIds = []
        self.sentIds = []
        for img in self.images:
            img_id = img['imgid']
            self.img2filename[img_id] = img['filename'].split('.')[0]
            self.imgIds.append(img_id)
            self.img2sents[img_id] = img['sentids']
            self.img2enggt[img_id] = self.imgname2enggt.get(img['filename'], [])
            for i, sent in enumerate(img['sentences']):
                self.sentences[sent['sentid']] = (sent['tokens'], sent['raw'])
                self.sentIds.append(sent['sentid'])
                self.sentId2imgId[sent['sentid']] = img_id
                sid = img['filename'].split('.')[0]+'#'+str(i)

        self.use_att = use_att
        if self.use_att == True:
            self.vf_dir = vf_dir
        else:
            self.vf_dir = vf_dir
            self.vf_reader = BigFile(vf_dir)

    def __getitem__(self, index):
        sentid = self.sentIds[index]
        img_id = self.sentId2imgId[sentid]
        img_name = self.img2filename[img_id]
        tokens, raw = self.sentences[sentid]
        caption = []
        # caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        caption = torch.Tensor(caption)
        img_captions = []
        for x in self.img2sents[img_id]:
            tokens, raw = self.sentences[x]
            temp = []
            # temp.append(self.vocab('<start>'))
            temp.extend([self.vocab(token) for token in tokens])
            temp.append(self.vocab('<end>'))
            img_captions.append(temp)

        eng_gt = self.img2enggt[img_id]

        if self.use_att == False:
            feature = np.array(self.vf_reader.read_one(img_name), dtype='float32')
            feature = torch.from_numpy(feature)
            return caption, feature, None, img_id, img_name, img_captions, eng_gt
        else:
            feature = np.load(os.path.join(self.vf_dir+'_fc', str(img_id)) + '.npy')
            att_feature = np.load(os.path.join(self.vf_dir + '_att', str(img_id)) + '.npz')['feat']
            feature = torch.from_numpy(feature)
            att_feature = torch.from_numpy(att_feature)
            return caption, feature, att_feature, img_id, img_name, img_captions, eng_gt

    def __len__(self):
        return len(self.sentIds)


def collate_fn(data):

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)

    captions, fc_feats, att_feats, img_ids, img_names, img_captions, eng_gts = zip(*data)
    
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    mask = targets > 0

    fc_feats = torch.stack(fc_feats, 0)
    if att_feats[0] is None:
        att_feats = None
    else:
        att_feats = torch.stack(att_feats, 0)
    data = {'fc_feats': fc_feats, 'att_feats': att_feats, 'targets':targets, 
            'lengths':lengths, 'mask':mask, 
            'img_ids':img_ids, 'img_names':img_names,
            'img_captions': img_captions, 'eng_gts':eng_gts}
    return data
 
    '''
        return images, targets, lengths, mask, img_ids, img_captions
    else:
        att_feats = torch.stack(att_feats, 0)
        return images, att_feats, targets, lengths, mask, img_id, img_captions
    '''


def get_loader(opt, vocab, split):

    input_json_path = path_util.get_input_json(opt.collection, split)
    vf_dir = path_util.get_feat_dir(opt.collection+split, opt.vf_name)
    cls_weight = opt.cross_lingual_similarity
    if cls_weight > 0:
        eng_gt_file = path_util.get_eng_gt_file(opt.collection)
    else:
        eng_gt_file = None

    if split == 'test':
        opt.shuffle = False
        
    coco = CocoDataset(input_json_path, vocab, vf_dir, 
                        use_att=opt.use_att, 
                        eng_gt_file=eng_gt_file,
                        rootpath=opt.rootpath)
    print(input_json_path, len(coco))
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                    batch_size=opt.batch_size,
                                    shuffle=opt.shuffle,
                                    num_workers=opt.num_workers,
                                    collate_fn=collate_fn)

    return data_loader