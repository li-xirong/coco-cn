
from __future__ import print_function

import sys
import os
import logging

from search_fusion_weight import load_config
from constant import *

formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s"
logging.basicConfig(
    format = formatter_log,
    datefmt = '%d %b %H:%M:%S')

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection fusion_config""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--zh_trainCollection", default="cococntrain", type="string", help="Chinese training set (default: cococntrain)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1

    testCollection = args[0]
    configName = args[1]
    config = load_config('fusion_configs/%s.py' % configName)
    rankMethod_list = config.run_list + [configName]
    for rankMethod in rankMethod_list:
        res_file = os.path.join(options.rootpath, testCollection, 'SimilarityIndex', options.zh_trainCollection, rankMethod, 'mir.txt')
        line = open(res_file).read().strip()
        print (rankMethod, line)

if __name__ == '__main__':
    main()

