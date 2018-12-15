from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import numpy as np
import logging

from constant import *
import utility

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


def process(options, collection, concept_file, tagging_method, new_feat_name):
    rootpath = options.rootpath
    overwrite = options.overwrite
    feat_dir = os.path.join(rootpath, collection, 'FeatureData', new_feat_name)

    if os.path.exists(feat_dir) and not overwrite:
        logger.info('%s exists. quit', feat_dir)
        return 0

    concepts = map(str.strip, open(concept_file).readlines())
    feat_dim = len(concepts)
    concept2index = dict(zip(concepts, range(feat_dim)))

    imset = []
    cached = set()
    data_file = os.path.join(rootpath, collection, 'autotagging', collection, tagging_method, 'id.tagvotes.txt')

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    fw = open(os.path.join(feat_dir,'feature.bin'), 'wb')

    count_line = 0

    for line in open(data_file):
        count_line += 1
        elems = line.strip().split()
        _id = elems[0]
        if _id in cached:
            continue
        cached.add(_id)
        del elems[0]
        vec = [0] * feat_dim
        for i in range(0, len(elems), 2):
            index = concept2index[elems[i]]
            vec[index] = float(elems[i+1])
        vec = np.array(vec, dtype=np.float32)
        vec.tofile(fw)
        imset.append(_id)
    fw.close()

    assert(len(imset) == len(set(imset)))
    fw = open(os.path.join(feat_dir, 'id.txt'), 'w')
    fw.write(' '.join(imset))
    fw.close()
    fw = open(os.path.join(feat_dir,'shape.txt'), 'w')
    fw.write('%d %d' % (len(imset), feat_dim))
    fw.close() 
    logger.info('%d lines parsed, %d unique ids' % (count_line, len(imset)))



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection concept_file tagging_method new_feat_name""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2], args[3])


if __name__ == "__main__":
    sys.exit(main())
