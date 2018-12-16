import sys
import os
import numpy as np
from basic.common import makedirsforfile, checkToSkip, printStatus
from bigfile import BigFile

INFO = __file__

def process(options, feat_dir, imsetfile, result_dir):

    resultfile = os.path.join(result_dir, 'feature.bin')
    if checkToSkip(resultfile, options.overwrite):
        sys.exit(0)

    imset = map(str.strip, open(imsetfile).readlines())
    print "requested", len(imset)

    feat_file = BigFile(feat_dir)
    
    makedirsforfile(resultfile)
    fw = open(resultfile, 'wb')

    done = []
    start = 0
  
    while start < len(imset):
        end = min(len(imset), start + options.blocksize)
        printStatus(INFO, 'processing images from %d to %d' % (start, end-1))
        toread = imset[start:end]
        if len(toread) == 0:
            break
        renamed, vectors = feat_file.read(toread)
        for vec in vectors:
            vec = np.array(vec, dtype=np.float32)
            vec.tofile(fw)
        done += renamed
        start = end
    fw.close()

    assert(len(done) == len(set(done)))
    with open(os.path.join(result_dir, 'id.txt'), 'w') as fw:
        fw.write(' '.join(done))
        fw.close()
    
    with open(os.path.join(result_dir,'shape.txt'), 'w') as fw:
        fw.write('%d %d' % (len(done), feat_file.ndims))
        fw.close()
    print '%d requested, %d obtained' % (len(imset), len(done))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] feat_dir imsetfile result_dir""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--blocksize", default=1000, type="int", help="nr of feature vectors loaded per time (default: 1000)")
    

    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1

    return process(options, args[0], args[1], args[2])

if __name__ == "__main__":
    sys.exit(main())

