import os
from basic.common import ROOT_PATH

def get_input_json(collection, split, rootpath=ROOT_PATH):
    name=collection+split
    input_json = os.path.join(rootpath, name, 'TextData', 'dataset_%s.json'%name)
    return input_json

def get_vocab(collection, rootpath=ROOT_PATH):
    name = collection+'train'
    vocab_file = os.path.join(rootpath, name, 'TextData', 'vocab_%s.pkl'%name)
    return vocab_file

def get_merged_vocab(collection, rootpath=ROOT_PATH):
    name = collection+'train'
    vocab_file = os.path.join(rootpath, name, 'TextData', 'merged_vocab.pkl')
    return vocab_file
    
def cached_tokens(collection, rootpath=ROOT_PATH):
    cached_tokens = os.path.join(rootpath, collection+'train', 'TextData', '%strain-idxs.p'%collection)
    return cached_tokens

def get_feat_dir(collection, feature, rootpath=ROOT_PATH):
    return os.path.join(rootpath, collection, 'FeatureData', feature)

def get_sub_model_dir(opt):
    if opt.cross_lingual_similarity > 0:
        model_setting += '_cls%.2f'%opt.cross_lingual_similarity
    specify_id = 'no_specified_id'
    if len(opt.id) > 0:
        specify_id = str(opt.id)
     
    return os.path.join( opt.model_name, specify_id, 'feedback_%d'%opt.feedback_start_epoch, opt.vf_name, 'lr'+str(opt.learning_rate) + '_decay'+str(opt.learning_rate_decay_start))

def get_model_dir(opt, rootpath=ROOT_PATH):
    #if opt.train_collection != None:
    #    return os.path.join(rootpath, opt.train_collection, 'Models', get_sub_model_dir(opt))
    return os.path.join(rootpath, opt.collection+'train', 'Models', get_sub_model_dir(opt))

def get_best_model(opt, rootpath=ROOT_PATH):
    #if opt.model_path:
    #    return os.path.join(opt.model_path, 'model-best.pth')
    return os.path.join(get_model_dir(opt, rootpath), 'model-best.pth')

def get_best_model_info(opt, rootpath=ROOT_PATH):
    #if opt.model_path:
    #    return os.path.join(opt.model_path, 'infos-best.pth')
    return os.path.join(get_model_dir(opt, rootpath), 'infos-best.pkl')


def get_output_dir(opt, split, rootpath=ROOT_PATH):
    #if opt.test_collection != None:
    #    return os.path.join(rootpath, opt.test_collection, 'Autocap', split, opt.collection, get_sub_model_dir(opt), 'beamsize'+str(opt.beam_size))
    return os.path.join(rootpath, opt.collection+split, 'Autocap', get_sub_model_dir(opt), 'beamsize'+str(opt.beam_size))

def get_anno_file(collection, split=None, rootpath=ROOT_PATH):
    name = collection #+split
    anno_file = os.path.join(rootpath, name, 'TextData', 'captions_%s.json'%name)
    return anno_file

def get_eng_gt_file(collection, rootpath=ROOT_PATH):
    name = collection 
    eng_file = os.path.join(rootpath, name, 'TextData', 'eng_dataset_%s.json'%name)
    return eng_file

def get_test_anno_file(collection, rootpath=ROOT_PATH):
    test_anno_file = os.path.join(rootpath, collection+'test', 'TextData', 'merged_captions_coco-cn_test.json')
    return test_anno_file
