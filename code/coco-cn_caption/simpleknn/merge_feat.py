import os, sys, array
import numpy as np

from basic.constant import ROOT_PATH
from basic.common import checkToSkip, makedirsforfile, printStatus
from basic.util import readImageSet

INFO = __file__

def process(options, feature, srcCollections, newCollection):
    assert(type(srcCollections) == list)
    
    temp = []
    [x for x in srcCollections if x not in temp and temp.append(x)] # unique source collections
    srcCollections = temp
    
    rootpath = options.rootpath
    
    resfile = os.path.join(rootpath, newCollection, 'FeatureData', feature, 'feature.bin')
    if checkToSkip(resfile, options.overwrite):
        return 0
    
    querysetfile = os.path.join(rootpath, newCollection, 'ImageSets', '%s.txt' % newCollection)
    try:
        query_set = set(map(str.strip, open(querysetfile).readlines()))
        printStatus(INFO, '%d images wanted' % len(query_set))
    except IOError:
        printStatus(INFO, 'failed to load %s, will merge all features in %s' % (querysetfile, ';'.join(srcCollections)))
        query_set = None
    
    makedirsforfile(resfile)
    fw = open(resfile, 'wb')
    printStatus(INFO, 'writing results to %s' % resfile)
    seen = set()
    newimset = []
    
    for collection in srcCollections:
        feat_dir = os.path.join(rootpath, collection, 'FeatureData', feature)
        with open(os.path.join(feat_dir, 'shape.txt')) as fr:
            nr_of_images, feat_dim = map(int, fr.readline().strip().split())
            fr.close()
        
        srcimset = open(os.path.join(feat_dir,'id.txt')).readline().strip().split()
        res = array.array('f')
        fr = open(os.path.join(feat_dir,'feature.bin'), 'rb')
        
        for i,im in enumerate(srcimset):
            res.fromfile(fr, feat_dim)
            if im not in seen:
                seen.add(im)
                if not query_set or im in query_set:
                    vec = res
                    vec = np.array(vec, dtype=np.float32)
                    vec.tofile(fw)
                    newimset.append(im)
            del res[:]
            if i%1e5 == 0:
                printStatus(INFO, '%d parsed, %d obtained' % (len(seen), len(newimset)))
        fr.close()       
        printStatus(INFO, '%d parsed, %d obtained' % (len(seen), len(newimset)))
                        
    fw.close()
    printStatus(INFO, '%d parsed, %d obtained' % (len(seen), len(newimset)))
    
    idfile = os.path.join(os.path.split(resfile)[0], 'id.txt')
    with open(idfile, 'w') as fw:
        fw.write(' '.join(newimset))
        fw.close()
        
    shapefile = os.path.join(os.path.split(resfile)[0], 'shape.txt')
    with open(shapefile, 'w') as fw:
        fw.write('%d %d' % (len(newimset), feat_dim))
        fw.close()



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] feature srcCollections  newCollection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath where data are stored (default: %s)" % ROOT_PATH)
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    return process(options, args[0], args[1].split(','), args[2])
    

if __name__ == "__main__":
    sys.exit(main())
