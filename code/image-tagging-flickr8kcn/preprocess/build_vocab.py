from __future__ import print_function

import sys, os
from util import checkToSkip, makedirsforfile

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')


def process(options, collection):
    rootpath = options.rootpath
    overwrite = options.overwrite
    topk = options.topk

    label_file = os.path.join(rootpath, collection, 'TextData', '%s.imglabel.txt' % collection)
    vocab_file = os.path.join(rootpath, collection, 'Annotations', 'concepts%s%d.txt' % (collection,topk))

    if checkToSkip(vocab_file, overwrite):
        return 0
   
    tag2count = {}

    for line in open(label_file):
        elems = line.strip().split()
        del elems[0] # because this is imgid
        for x in elems:
            tag2count[x] = tag2count.get(x,0) + 1

    taglist = sorted(tag2count.iteritems(), key=lambda v:(v[1],v[0]), reverse=True)
    assert(len(taglist)>=topk)
    taglist = taglist[:topk]
    makedirsforfile(vocab_file)
    fw = open(vocab_file, 'w')
    fw.write('\n'.join([x[0] for x in taglist]))
    fw.close()
    logger.info('A vocabulary of %d labels at %s', len(taglist), vocab_file)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)"%ROOT_PATH)
    parser.add_option("--topk", default=512, type="int", help="topk (default: 512)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    return process(options, args[0])

  
if __name__=="__main__":
    sys.exit(main())

