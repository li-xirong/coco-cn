import sys
import os

from bigfile import BigFile

os.system('python merge_feat.py f3d toydata,toydata,toydata,toydata2 newdata --rootpath ./ --overwrite 1')

feat_file = BigFile('newdata/FeatureData/f3d')
renamed, vectors = feat_file.read(feat_file.names)

for _id, _vec in zip(renamed, vectors):
    print _id, _vec
