from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
from dataset import LabelSet
import utility
from constant import *

import numpy as np
import logging

logger = logging.getLogger(__file__)
formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s"
logging.basicConfig(
    format = formatter_log,
    datefmt = '%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


class Eval:
    def __init__(self, collection, concept_file, rootpath=ROOT_PATH):
        self.label_set = LabelSet(collection, concept_file, rootpath=rootpath)

    def score(self, pred_file, verbose=1):
        logger.info('scoring %s', pred_file)
        img_ids = []
        top_indices = []
        for line in open(pred_file):
            elems = line.strip().split()
            _id = elems[0]
            if _id not in self.label_set.im2labels:
                continue
            del elems[0]
            ranked = [self.label_set.concept2index[elems[i]] for i in range(0, len(elems), 2)]
            top_indices.append(ranked)
            img_ids.append(_id)

        top_indices = np.array(top_indices)
        truth = self.label_set.get_label_matrix(img_ids)
        perf_table = utility.compute_hit_precision_recall_f1(top_indices, truth)

        output = []
        metrics = str.split('hit precision recall f_measure')
        ranks = [1, 5, 10]
        output.append('-'*50)
        output.append('rank %s' % ' '.join(metrics))
        for i,rank in enumerate(ranks):
            output.append('%4d %s' % (rank, ' '.join(['%.3f' % x for x in perf_table[i,:]])))
        output.append('-'*50)

        if verbose:
            print ('\n'.join(output))
               
        res_file = pred_file + '.eval'
        logger.info('save evaluation results to %s', res_file)
        open(res_file, 'w').write('\n'.join(output)) 
        return perf_table


def process(options, collection, concept_file, tagging_method):
    rootpath = options.rootpath
    pred_file = os.path.join(rootpath, collection, 'autotagging', collection, tagging_method, 'id.tagvotes.txt')
    evaluator = Eval(collection, concept_file, rootpath=rootpath)
    evaluator.score(pred_file)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection concept_file tagging_method""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)"%ROOT_PATH)

    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1

    return process(options, args[0], args[1], args[2])


if __name__=="__main__":
    sys.exit(main())


