
from __future__ import print_function

import sys
import os
import logging
import numpy as np

from constant import *
from eval_perf import cal_perf

formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s"
logging.basicConfig(
    format = formatter_log,
    datefmt = '%d %b %H:%M:%S')

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']

def read_sent_score_file(input_file, imageids, sentids, rank_norm=0):
    image2index = dict(zip(imageids, range(len(imageids))))
    sent2index = dict(zip(sentids, range(len(sentids))))
    nr_of_images = len(imageids)
    nr_of_sents = len(sentids)

    score_table = np.zeros( (nr_of_sents, nr_of_images) )

    for line in open(input_file).readlines():
        elems = line.strip().split()
        sent_id = elems[0].split('#')[0]
        del elems[0]
        assert(len(elems)%2 == 0)
        ranklist = [(elems[i].split('#')[0], float(elems[i+1])) for i in range(0, len(elems), 2)]
        assert(len(ranklist) == nr_of_images)
        ranklist.sort(key=lambda v:v[1], reverse=True)
        if rank_norm:
            ranklist = [(ranklist[i][0], 1.0-float(i)/nr_of_sents) for i in range(nr_of_images)]
        sent_index = image2index[sent_id]
        
        for (imageid, score) in ranklist:
            image_index = image2index[imageid]
            score_table[sent_index, image_index] = score

    return score_table 


def process(options, collection, fusion_config, config):
    rootpath = options.rootpath
    #overwrite = options.overwrite
    zh_trainCollection = options.zh_trainCollection

    res_dir = os.path.join(rootpath, collection, 'SimilarityIndex', zh_trainCollection)
    res_file = os.path.join(res_dir, fusion_config, 'sent.id.score.txt')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    #if os.path.exists(res_file) and not overwrite:
    #    logger.info('%s exists. quit', res_file)
    #    return res_file

    logger.info('%s', zip(config.run_list, config.weights))
    logger.info('perform rank-based normalization? %s', 'Yes' if config.rank_norm else 'No')
    
    imgid_file = os.path.join(rootpath, collection, 'ImageSets', collection+'.txt')
    imageids = [x.strip().split()[0] for x in open(imgid_file).readlines()]
    assert(len(imageids) == len(set(imageids)))
    sentids = imageids

    nr_of_images = len(imageids)
    nr_of_sents = len(sentids)
    nr_of_runs = len(config.run_list)

    new_score_table = np.zeros((nr_of_sents, nr_of_images))

    for run_index, run_name in enumerate(config.run_list):
        input_file = os.path.join(res_dir, run_name, 'sent.id.score.txt')
        logger.info('load scores from %s', input_file)
        score_table = read_sent_score_file(input_file, imageids, sentids, config.rank_norm)
        new_score_table += config.weights[run_index] * score_table

    logger.info('re-rank')
    top_hits = np.argsort(new_score_table)[:,::-1]
    
    if not os.path.exists(os.path.split(res_file)[0]):
        os.makedirs(os.path.split(res_file)[0])

    logger.info('save results to %s', res_file)    
    fw = open(res_file, 'w')
    for i, sent_id in enumerate(sentids):
        top_scores = new_score_table[i, top_hits[i]]
        ranklist = [(imageids[x], y) for (x, y) in zip(top_hits[i], top_scores)]
        fw.write('%s %s\n' % (sent_id, ' '.join(['%s %g' % (x[0],x[1]) for x in ranklist])))
    fw.close()
    return res_file

def performance(score_file):
    recall_name, recall_score, med_r, mean_r, mean_invert_r = cal_perf(score_file)
    return mean_invert_r

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] searchCollection testCollection fusion_config""")

    #parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--zh_trainCollection", default="cococntrain", type="string", help="Chinese training set (default: cococntrain)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1

    searchCollection = args[0]
    testCollection = args[1]
    zh_trainCollection = options.zh_trainCollection

    config = load_config('fusion_configs/%s.py' % args[2])
    res_dir = os.path.join(options.rootpath, searchCollection, 'SimilarityIndex', zh_trainCollection)
    res_file = os.path.join(res_dir, args[2], 'fusion_weight_score.txt')
    if not os.path.exists(os.path.split(res_file)[0]):
        os.makedirs(os.path.split(res_file)[0])
    weights = [0.01*i for i in range(100)]
    results = []
    for i in weights:
        config.weights = [i, 1-i]
        score_file = process(options, searchCollection, args[2], config)
        mir = performance(score_file)
        results.append((i, 1-i, mir))

    fw = open(res_file, 'w')
    results = sorted(results, key = lambda v:v[2], reverse=True)
    for r in results:
        fw.write('{} {} {}\n'.format(r[0], r[1], r[2]))
    fw.close()
    # test
    test_res_dir = os.path.join(options.rootpath, testCollection, 'SimilarityIndex', zh_trainCollection)
    test_res_file = os.path.join(test_res_dir, args[2], 'mir.txt')
    if not os.path.exists(os.path.split(test_res_file)[0]):
        os.makedirs(os.path.split(test_res_file)[0])
    fw = open(test_res_file, 'w')
    config.weights = [results[0][0], results[0][1]]
    score_file = process(options, testCollection, args[2], config)
    mir = performance(score_file)

    #print('mir: ', mir)
    #fw.write('mir: {}\n'.format(mir))

    print('mean_inverted_rank: ', round(mir, 3))
    fw.write('mean_inverted_rank: {}\n'.format(round(mir, 3)))
    fw.close()


if __name__ == "__main__":
    sys.exit(main())

