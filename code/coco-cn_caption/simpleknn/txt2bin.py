'''
# convert one or multiple feature files from txt format to binary (float32) format
'''

import os, sys, math
import numpy as np
from optparse import OptionParser


def checkToSkip(filename, overwrite):
    if os.path.exists(filename):
        print ("%s exists." % filename),
        if overwrite:
            print ("overwrite")
            return 0
        else:
            print ("skip")
            return 1
    return 0


def process(feat_dim, inputTextFiles, resultdir, overwrite):
    res_binary_file = os.path.join(resultdir, 'feature.bin')
    res_id_file = os.path.join(resultdir, 'id.txt')

    if checkToSkip(res_binary_file, overwrite):
        return 0

    if os.path.isdir(resultdir) is False:
        os.makedirs(resultdir)

    fw = open(res_binary_file, 'wb')
    processed = set()
    imset = []
    count_line = 0
    failed = 0

    for filename in inputTextFiles:
        print ('>>> Processing %s' % filename)
        for line in open(filename):
            count_line += 1
            elems = line.strip().split()
            if not elems:
                continue
            name = elems[0]
            if name in processed:
                continue
            processed.add(name)

            del elems[0]
            vec = np.array(map(float, elems), dtype=np.float32)
            okay = True
            for x in vec:
                if math.isnan(x):
                    okay = False
                    break
            if not okay:
                failed += 1
                continue
          
            assert(len(vec) == feat_dim), "dimensionality mismatch: required %d, input %d, id=%s, inputfile=%s" % (feat_dim, len(vec), name, filename)
            vec.tofile(fw)
            #print name, vec
            imset.append(name)
    fw.close()

    fw = open(res_id_file, 'w')
    fw.write(' '.join(imset))
    fw.close()
    fw = open(os.path.join(resultdir,'shape.txt'), 'w')
    fw.write('%d %d' % (len(imset), feat_dim))
    fw.close() 
    print ('%d lines parsed, %d ids,  %d failed ->  %d unique ids' % (count_line, len(processed), failed, len(imset)))



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = OptionParser(usage="""usage: %prog [options] nDims inputTextFile isFileList resultDir""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1

    fea_dim = int(args[0])
    inputTextFile = args[1]
    if int(args[2]) == 1:
        inputTextFiles = [x.strip() for x in open(inputTextFile).readlines() if x.strip() and not x.strip().startswith('#')]
    else:
        inputTextFiles = [inputTextFile]
    return process(fea_dim, inputTextFiles, args[3], options.overwrite)

if __name__ == "__main__":
    sys.exit(main())




