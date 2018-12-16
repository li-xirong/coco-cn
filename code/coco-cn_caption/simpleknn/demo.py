import sys
import os

import simpleknn
from bigfile import BigFile

if __name__ == "__main__":
    rootpath = './'
    trainCollection = 'toydata'
    nimages = 2
    feature = 'f1'
    dim = 3

    testCollection = trainCollection
    testset = testCollection
   
    featureDir = os.path.join(rootpath, trainCollection, "FeatureData", feature)
    searcher = simpleknn.load_model(os.path.join(featureDir, "feature.bin"), dim, nimages, os.path.join(featureDir, "id.txt"))
    searcher.set_distance('l2')
    searcher.set_distance('l1')
    print ("[simpleknn] dim=%d, nr_images=%d" % (searcher.get_dim(), searcher.get_nr_images()))


    testfeaturedir = os.path.join(rootpath, testCollection, 'FeatureData', feature)
    testfeaturefile = BigFile(testfeaturedir, dim)
    testset = testfeaturefile.names

    for testid in testset:
        testfeature = testfeaturefile.read_one(testid)
        visualNeighbors = searcher.search_knn(testfeature, max_hits=20000)
        print testid, len(visualNeighbors), " ".join(["%s %.3f" % (v[0],v[1]) for v in visualNeighbors[:3]])

 


