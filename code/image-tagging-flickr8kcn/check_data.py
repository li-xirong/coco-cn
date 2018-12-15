from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

from tf_tagging.constant import ROOT_PATH

rootpath = ROOT_PATH
collections = 'flickr8k flickr8ktrain flickr8kval flickr8ktest flickr8kcn flickr8kcntrain flickr8kcnval flickr8kcntest flickr30k flickr30ktrain flickr30kval flickr30ktest'.split()

class TestSuite (unittest.TestCase):

    def test_rootpath(self):
        self.assertTrue(os.path.exists(rootpath))

    def test_img_feat_files(self):
        feat_name = 'pyresnext-101_rbps13k,flatten0_output,osl2'
        for collection in collections:
            shape_file = os.path.join(rootpath, collection, 'FeatureData', feat_name, 'shape.txt')
            self.assertTrue(os.path.exists(shape_file), '%s missing' % shape_file)

    def test_imset_files(self):
        for collection in 'flickr8k flickr8kcn flickr30k'.split():
            for dataset in 'train val test'.split():
                sub_collection = '%s%s' % (collection, dataset)
                imset_file = os.path.join(rootpath, sub_collection, 'ImageSets', '%s.txt'%sub_collection)
                self.assertTrue(os.path.exists(imset_file), '%s missing' % imset_file)

    def test_sent_files(self):
        for collection in 'flickr8k flickr8kcn flickr30k'.split():
            pos_name = 'boson' if collection.endswith('cn') else 'stanford'
            pos_sent_file = os.path.join(rootpath, collection, 'TextData', '%s.%spos.txt' % (collection,pos_name))
            self.assertTrue(os.path.exists(pos_sent_file), '%s missing' % pos_sent_file)

suite = unittest.TestLoader().loadTestsFromTestCase(TestSuite)
unittest.TextTestRunner(verbosity=2).run(suite)
