# coding: utf8
# performance metric of image-to-text retrieval
# R@K (K = 1, 5, 10), Median rank (Med r) and Mean rank (Mean r)
# In particular, R@K computes the percentage of test images for which at least one correct result is found among the top-K retrieved sentences,
# Med r, Mean r is the median, mean rank of the first correct result in the ranking list respectively. 
# Hence, higher R@K and lower Med r, Mean r means better performance.
#
import os
import sys
import numpy as np
from basic.common import printStatus

INFO = os.path.basename(__file__)

class MetricScorer:

    def __init__(self, k=0):
        self.k = k

    def score(self, sorted_labels):
        return 0.0

    def getLength(self, sorted_labels):
        length = self.k
        if length>len(sorted_labels) or length<=0:
            length = len(sorted_labels)
        return length

    def name(self):
        if self.k > 0:
            return "%s@%d" % (self.__class__.__name__.replace("Scorer",""), self.k)
        return self.__class__.__name__.replace("Scorer","")


class RecallScorer (MetricScorer):

    def score(self, sorted_labels):
        length = self.getLength(sorted_labels)
        for i in xrange(length):
            if 1 <= sorted_labels[i]:
                return 1.0
        return 0.0


# image-to-text retrieval
def cal_perf(prediction_file, verbose = 1):
    '''
    if prediction_file.find('setA') != -1:
        fix = '#1'
    elif prediction_file.find('setB') != -1:
        fix = '#2'
    '''
    scorers = [RecallScorer(k) for k in [1, 5, 10]]

    res = [0] * len(scorers)
    nr_of_images = 0
    nr_of_sents = 0
    
    first_matched_idexs =[]
    for line in open(prediction_file):
        elems = line.strip().split()
        imageid = elems[0]
        del elems[0]

        assert(len(elems)%2 == 0)
        sentids = elems[::2]
        nr_of_sents = len(sentids)
        hit_list = []
        flag = 1
        for i in range(len(sentids)):
            if  sentids[i].split('#')[0] == imageid.split('#')[0]:
                hit_list.append(1)
                if flag == 1:
                    first_matched_idexs.append(i+1)
                    flag = 0
            else:
                hit_list.append(0)
            if len(hit_list) > 20  and flag == 0:
                break

        hit_list = hit_list[:20] # consider at most the first 20 predicted tags

        perf = [scorer.score(hit_list) for scorer in scorers]
        res = [res[i] + perf[i] for i in range(len(scorers))]
        nr_of_images += 1

    printStatus(INFO, 'nr of images: %d' % nr_of_images)
    printStatus(INFO, 'nr of sentents: %d' % nr_of_sents)
    res = [x/nr_of_images for x in res]

    #print('first_matched_idexs: ', len(first_matched_idexs))
    recall_name = ' '.join([x.name() for x in scorers])
    recall_score = ' '.join(['%.3f' % x for x in res])
    #assert len(first_matched_idexs) == nr_of_images    
    med_r = sorted(first_matched_idexs)[nr_of_images/2-1 ]
    mean_r = np.mean(first_matched_idexs)

    mean_invert_r = []
    for i in first_matched_idexs:
        mean_invert_r.append(1.0/i)
    mean_invert_r = np.mean(mean_invert_r)


    if verbose == 1:
        print recall_name
        print recall_score
        print 'Med r: ', med_r
        print 'Mean r: ', mean_r
        print 'mean inverted r: ', round(mean_invert_r, 3)

    return (recall_name, recall_score, med_r, mean_r, mean_invert_r)


# text-to-image retrieval
def cal_perf_t2i(prediction_file, verbose=1):
    scorers = [RecallScorer(k) for k in [1, 5, 10]]

    res = [0] * len(scorers)
    nr_of_sents = 0
    nr_of_images = 0

    first_matched_idexs = []
    for line in open(prediction_file):
        elems = line.strip().split()
        sentid = elems[0]
        del elems[0]

        assert (len(elems) % 2 == 0)
        imageids = elems[::2]
        nr_of_images = len(imageids)
        hit_list = []
        flag = 1
        for i in range(len(imageids)):
            if sentid.find(imageids[i]) == 0:
                hit_list.append(1)
                if flag == 1:
                    first_matched_idexs.append(i + 1)
                    flag = 0
            else:
                hit_list.append(0)
            if len(hit_list) > 20 and flag == 0:
                break

        hit_list = hit_list[:20]  # consider at most the first 20 predicted tags

        perf = [scorer.score(hit_list) for scorer in scorers]
        res = [res[i] + perf[i] for i in range(len(scorers))]

    printStatus(INFO, 'nr of sentences: %d' % nr_of_sents)
    printStatus(INFO, 'nr of images: %d' % nr_of_images)
    res = [x / nr_of_images for x in res]

    recall_name = ' '.join([x.name() for x in scorers])
    recall_score = ' '.join(['%.3f' % x for x in res])

    assert len(first_matched_idexs) == nr_of_images
    med_r = sorted(first_matched_idexs)[nr_of_images / 2 - 1]
    mean_r = np.mean(first_matched_idexs)

    mean_invert_r = []
    for i in first_matched_idexs:
        mean_invert_r.append(1.0/i)
    mean_invert_r = np.mean(mean_invert_r)

    if verbose == 1:
        print recall_name
        print recall_score
        print 'Med r: ', med_r
        print 'Mean r: ', mean_r
        print 'mean inverted r: ', round(mean_invert_r, 3)

    return (recall_name, recall_score, med_r, mean_r, mean_invert_r)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser

    parser = OptionParser(usage="""usage: %prog [options] prediction_file""")
    # parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    # parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    # parser.add_option("--metric", type="string", default="recall",  help="performance metric, namely recall")
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    #assert (options.metric in ['recall'])   

    return cal_perf(args[0])



if __name__=="__main__":
    sys.exit(main())

