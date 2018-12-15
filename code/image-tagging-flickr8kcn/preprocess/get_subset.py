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


def process(options, collection, src_file, res_file):
    rootpath = options.rootpath
    overwrite = options.overwrite

    imset_file = os.path.join(rootpath, collection, 'ImageSets', '%s.txt' % collection)
    imset = set(map(str.strip, open(imset_file).readlines()))

    if checkToSkip(res_file, overwrite):
        return 0
 
    cached = set()  
    obtained = 0

    makedirsforfile(res_file)
    fw = open(res_file, 'w')

    for line in open(src_file):
        imgid = line.split()[0]
        if imgid in cached:
            continue
        cached.add(imgid)
        if imgid in imset:
            fw.write(line)
            obtained += 1
    fw.close()
    logger.info('%d wanted, %d obtained', len(imset), obtained)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection src_file res_file""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)"%ROOT_PATH)

    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1

    return process(options, args[0], args[1], args[2])

  
if __name__=="__main__":
    sys.exit(main())

