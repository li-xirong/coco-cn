import os, random

import simpleknn
from bigfile import BigFile

rootpath = './'
collection = 'newdata'
feature = 'f3d'
feat_dim = 3

feat_dir = '%s/FeatureData/%s' % (collection, feature)
txt_file = os.path.join(feat_dir, 'id.feature.txt')
os.system('python txt2bin.py %d %s 0 %s --overwrite 1' % (feat_dim, txt_file, feat_dir))

for p in [1,2]:
    os.system('python norm_feat.py %s --p %d --overwrite 1' % (feat_dir, p))

    
for feature in str.split('f3d f3dl1 f3dl2'):
    feat_dir = os.path.join(rootpath,collection,'FeatureData',feature)
    id_file = os.path.join(feat_dir, "id.txt")
    with open(os.path.join(feat_dir, 'shape.txt')) as fr:
        nr_of_images, dim = map(int, fr.readline().split())
        fr.close()
    
    feat_file = BigFile(feat_dir)
    imset = feat_file.names

    searcher = simpleknn.load_model(os.path.join(feat_dir, "feature.bin"), dim, nr_of_images, id_file)
    renamed,vectors = feat_file.read(imset)

    for name,vec in zip(renamed,vectors):
        for distance in ['l1', 'l2']:
            searcher.set_distance(distance)
            visualNeighbors = searcher.search_knn(vec, max_hits=100)
            print name, feature, distance, visualNeighbors[:3]
