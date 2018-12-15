from __future__ import print_function

import sys, os
from nltk.stem import WordNetLemmatizer
from util import checkToSkip

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')
STOPWORDS = set(map(str.strip, open('stopwords.txt').readlines()))
POS_SET = {'en':set(['J', 'V', 'N']), 'zh':set(['a', 'v', 'n'])}
lemmatizer = WordNetLemmatizer()


def process(options, collection):
    rootpath = options.rootpath
    lang = options.lang
    overwrite = options.overwrite

    res_file = os.path.join(rootpath, collection, 'TextData', '%s.imglabel.txt' % collection)
    if checkToSkip(res_file, overwrite):
        return 0
   
    pos_method = 'stanford' if 'en' == lang else 'boson'
    pos_sent_file = os.path.join(rootpath, collection, 'TextData', '%s.%spos.txt' % (collection,pos_method))
    
    dict_img = {}

    for line in open(pos_sent_file):
        elems = line.strip().split()
        imgid = elems[0].split('#')[0]

        if imgid not in dict_img:
            dict_img[imgid] = {}

        for x in elems[1:]:
            if len(x.split(':')) != 2:
                #logger.error('invalid format %s', x)
                continue
            word, pos = x.split(':')
            if pos[0] in POS_SET[lang] and word not in STOPWORDS:
                if 'en' == lang:
                    if pos[0] == 'N':
                        word = lemmatizer.lemmatize(word)
                    elif pos[0] == 'V':
                        word = lemmatizer.lemmatize(word, pos='v')
                    else:
                        continue
                else:
                   if pos[0] != 'n' and len(word)<=3:
                       #print (len(word), word)
                       continue
                dict_img[imgid][word] = dict_img[imgid].get(word, 0) + 1


    fw = open(res_file, 'w')

    label_stat = []
    zero = 0
    for imgid, word2count in dict_img.iteritems():
        labels = [word for word,c in word2count.iteritems() if c>= 2]
        fw.write(' '.join([imgid] + labels) + '\n')
        label_stat.append(len(labels))
        if len(labels) == 0:
            logger.info('image %s has no label', imgid)
            zero += 1
    fw.close()
    logger.info('number of images with zero label: %d', zero)
    logger.info('number of labels per image: min (%d), max (%d), mean (%.1f)', min(label_stat), max(label_stat), sum(label_stat)/float(len(label_stat)))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)"%ROOT_PATH)
    parser.add_option("--lang", default="en", type="string", help="language (default: en)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    return process(options, args[0])


  
if __name__=="__main__":
    sys.exit(main())

