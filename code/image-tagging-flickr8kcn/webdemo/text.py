import sys
import os

def get_sent(collection, rootpath):
    datafile = os.path.join(rootpath, collection, 'results_20130124.token')
    im2sents = {}
    
    for line in open(datafile):
        name, sent = line.strip().split('\t')
        imageid = name.split('#')[0][:-4]
        im2sents.setdefault(imageid, []).append(sent)
    return im2sents

    
def get_tags(collection, rootpath):
    datafile = os.path.join(rootpath, collection, 'TextData', 'id.userid.rawtags.txt')
    im2tags = {}
    
    for line in open(datafile):
        imageid, userid, tags = line.strip().split('\t')
        im2tags[imageid] = tags.split()
    return im2tags
 
    