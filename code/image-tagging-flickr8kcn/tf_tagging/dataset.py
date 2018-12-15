from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import logging

from constant import *
from bigfile import BigFile
from utility import get_concept_file

logger = logging.getLogger(__file__)
formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s"
logging.basicConfig(
    format = formatter_log,
    datefmt = '%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


class LabelSet:
    def __init__(self, collection, concept_file, rootpath=ROOT_PATH):
        #concept_file = os.path.join(rootpath, collection, 'Annotations', annotationName)
        self.concepts = map(str.strip, open(concept_file).readlines())
        self.concept2index = dict(zip(self.concepts, range(len(self.concepts))))
        assert(len(self.concepts) == len(self.concept2index))
        self.num_labels = len(self.concept2index)
        
        #annotation_file = concept_file + '_image_labels'
        annotation_file = os.path.join(rootpath, collection, 'TextData', '%s.imglabel.txt'%collection)
        self.im2labels = {}
       
        # image-category pairs
        all_data = open(annotation_file).readlines()
        for line in all_data:
            data = line.strip().split()
            if len(data) <= 1:
                continue
            img_id = data[0]
            self.im2labels[img_id] = [self.concept2index[x] for x in data[1:] if x in self.concept2index]
       
        logger.info('%d images, %d concepts', len(self.im2labels), self.num_labels)

 
    def get_label_vector(self, img_id):
        labels = self.im2labels[img_id]
        vec = [0] * self.num_labels
        for lab in labels:
            vec[lab] = 1
        return vec


    def get_label_matrix(self, img_ids):
        Y = np.zeros((len(img_ids), self.num_labels))
        for i in range(len(img_ids)):
           Y[i] = self.get_label_vector(img_ids[i])
        return Y



class DataBatchIterator(object):
    def __init__(self, collections, concept_files, feature, batch_size=100, rootpath=ROOT_PATH):
        assert(len(collections) == len(concept_files))
        self.batch_size = batch_size
        self.feat_file = BigFile(os.path.join(rootpath, collections[0], 'FeatureData', feature))
        self.label_set = LabelSet(collections[0], concept_files[0], rootpath)
        self.aux_label_set = None

        if len(collections) > 1:
            self.aux_label_set = LabelSet(collections[1], concept_files[1], rootpath)

        self.img_ids = sorted(self.label_set.im2labels.keys())
        self.num_labels = self.label_set.num_labels
        self.aux_num_labels = self.aux_label_set.num_labels if self.aux_label_set else 0
        self.update()

    def update(self):
        self.num_samples = len(self.img_ids)
        self.num_batches =  int(np.ceil(self.num_samples/float(self.batch_size)))
        self.feat_dim = self.feat_file.ndims


    def shuffle(self):
        logger.info('dataset shuffle')
        random.shuffle(self.img_ids)

    def __iter__(self):
        n_samples = self.num_samples
        bs = self.batch_size

        for i in range((n_samples + bs - 1) // bs):
            start = i * bs 
            end = min(n_samples, start + bs)
            renamed, feats = self.feat_file.read(self.img_ids[start:end])
            Y = self.label_set.get_label_matrix(renamed) if self.label_set else None
            YE = self.aux_label_set.get_label_matrix(renamed) if self.aux_label_set else None
            yield (renamed, np.asarray(feats), Y, YE)


class TestDataBatchIterator(DataBatchIterator):
    def __init__(self, collection, feature, batch_size=100, rootpath=ROOT_PATH):
        self.feat_file = BigFile(os.path.join(rootpath, collection, 'FeatureData', feature))
        self.batch_size = batch_size
        self.label_set = None
        self.aux_label_set = None
        self.img_ids = map(str.strip, open(os.path.join(rootpath, collection, 'ImageSets', collection+'.txt')).readlines())
        self.update()


if __name__ == '__main__':
    rootpath = ROOT_PATH
    trainCollection = 'flickr8ktrain'
    valCollection = 'flickr8kval'
    annotationName = 'concepts%s512.txt' % trainCollection
    concept_file = get_concept_file(trainCollection, annotationName, rootpath)
    feature = 'pyresnext-101_rbps13k,flatten0_output,osl2'
    label_set = LabelSet(trainCollection, concept_file)

    vec = label_set.get_label_vector('2513260012_03d33305cf')
    print ([x for x in zip(vec, label_set.concepts) if x[0] == 1])

    data_iter = DataBatchIterator([trainCollection], [concept_file], feature, batch_size=1000)
    #data_iter.shuffle()
    for minibatch_index, (ids, X, Y, Ye) in enumerate(data_iter):
        print (minibatch_index, ids[:5], X.shape, Y.shape)

    print ('-'*100)

    data_iter = TestDataBatchIterator(valCollection, feature, batch_size=500)
    for minibatch_index, (ids, X, _, _) in enumerate(data_iter):
        print (minibatch_index, ids[:5], X.shape)

