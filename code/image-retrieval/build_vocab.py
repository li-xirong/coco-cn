# 
#  We convert all sentences to lowercase, discard non-alphanumeric characters. 
#  We filter words to those that occur at least $freq_threshold times in the training set.
#  Ref: Deep Visual-Semantic Alignments for Generating Image Descriptions  (Data Preprocessing)
#
from __future__ import print_function 
import os
import sys

from basic.constant import ROOT_PATH
from basic.common import checkToSkip, makedirsforfile
from utils.text import clean_str, clean_str_filter_stop
from utils.util import which_language

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


def process(options, collection, text_style, threshold):
    logger.info("processing %s ...", collection)
    rootpath = options.rootpath
    overwrite = options.overwrite
    threshold = int(threshold)
    lang = which_language(collection)

    input_file = os.path.join(rootpath, collection, 'TextData', '%s.caption.txt'%collection)
    output_vocab_file = os.path.join(rootpath, collection, 'TextData/vocabulary', text_style, 'word_vocab_%d.txt'%threshold)
    output_vocab_counter_file = os.path.join(rootpath, collection ,'TextData/vocabulary', text_style, 'word_vocab_counter_%d.txt'%threshold)

    if checkToSkip(output_vocab_file, overwrite):
        sys.exit(0)
    makedirsforfile(output_vocab_file)

    word2counter = {}
    for index, line in enumerate(open(input_file)):
        sid, sent = line.strip().split(" ", 1)
        if text_style == "bow":
            sent = clean_str(sent, lang)
        elif text_style == "bow_filterstop":
            sent = clean_str_filter_stop(sent, lang)
        if index == 0:
            logger.info(line.strip())
            logger.info('After processing: %s %s', sid, ' '.join(sent))
        for word in sent:
            word2counter[word] = word2counter.get(word, 0) + 1

    sorted_wordCounter = sorted(word2counter.iteritems(), key = lambda a:a[1], reverse=True)

    output_line_vocab = [ x[0] for x in sorted_wordCounter if x[1] >= threshold ]
    output_line_vocab_counter = [ x[0] + ' '  + str(x[1]) for x in sorted_wordCounter if x[1] >= threshold ]

    open(output_vocab_file, 'w').write('\n'.join(output_line_vocab))
    open(output_vocab_counter_file, 'w').write('\n'.join(output_line_vocab_counter))
    logger.info('A vocabulary of %d words has been built for %s', len(output_line_vocab), collection)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection text_style threshold""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)

    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    return process(options, args[0], args[1], args[2])

if __name__ == "__main__":
    sys.exit(main())
