# -*- coding:utf-8 -*-
import os
import sys
sys.path.append("..")
import json
import argparse

import basic.path_util as path_util
from basic.common import makedirsforfile
from basic.common import ROOT_PATH

def main(collection):
    input_json_path = path_util.get_input_json(collection, split='')
    with open(input_json_path) as f:
        dataset = json.loads(f.readline())
    images = dataset['images']
    splits = ['train', 'val', 'test']
    split_data = {}
    for s in splits:
        # split_data[s] = {'images':[], 'dataset':dataset['dataset']+s}
        split_data[s] = {'images':[]}
    scount = 0
    for i, image in enumerate(images):
        image['sentids'] = []
        for sent in image['sentences']:
            sent['sentid']=scount
            image['sentids'].append(scount)
            scount+=1
        split = image['split']
        if split in splits:
            split_data[split]['images'].append(image)
        else:
            if True: #use restval for training
                split_data['train']['images'].append(image)

    for s in splits:
        output_file = path_util.get_input_json(collection, split=s)
        makedirsforfile(output_file)
        with open(output_file, 'w') as f:
            print('Dumping json data file to %s'%output_file)
            json.dump(split_data[s], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    rootpath = ROOT_PATH
    parser.add_argument('--rootpath', default=rootpath, help='rootpath of the data')
    parser.add_argument('--collection', required=True, help='collection')
    args = parser.parse_args()
    collection = args.collection
    main(collection)
