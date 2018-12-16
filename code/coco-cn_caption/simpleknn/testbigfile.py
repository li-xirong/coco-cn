import os, random

import simpleknn
from bigfile import BigFile

rootpath = '/Users/xirong/VisualSearch'
collection = 'train10k'
nr_of_images = 10000
feature = 'color64'
dim = 64

feature_dir = os.path.join(rootpath,collection,'FeatureData',feature)
feature_file = BigFile(feature_dir, dim)
imset = map(str.strip, open(os.path.join(rootpath,collection,'ImageSets','%s.txt'%collection)).readlines())
imset = random.sample(imset, 10)

searcher = simpleknn.load_model(os.path.join(feature_dir, "feature.bin"), dim, nr_of_images, os.path.join(feature_dir, "id.txt"))
searcher.set_distance('l1')
renamed,vectors = feature_file.read(imset)

for name,vec in zip(renamed,vectors):
    visualNeighbors = searcher.search_knn(vec, max_hits=100)
    print name, visualNeighbors[:3]
