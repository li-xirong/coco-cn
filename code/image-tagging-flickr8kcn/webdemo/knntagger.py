import os
import operator


class TagVoteTagger:
    def __init__(self, collection, rootpath):
        self._load_tag_data(collection, tpp='lemm', rootpath=rootpath)
    
    def _load_tag_data(self, collection, tpp, rootpath):
        tagfile = os.path.join(rootpath, collection, "TextData", "id.userid.%stags.txt" % tpp)
        self.textstore = {}
        self.tag2freq = {}
        self.nr_of_images = 0

        if os.path.exists(tagfile):
            for line in open(tagfile):
                imageid, userid, tags = line.split('\t')
                tags = tags.lower()
                self.textstore[imageid] = (userid, tags)
                tagset = set(tags.split())
                for tag in tagset:
                    self.tag2freq[tag] = self.tag2freq.get(tag,0) + 1
                self.nr_of_images += 1
        print ('%d images, %d distinct tags' % (self.nr_of_images, len(self.tag2freq)))

        
    def tagprior(self, tag):
        return float(self.k) * self.tag2freq.get(tag,0) / self.nr_of_images
    
    
    
    def predict(self, neighbors):
        tag2vote = {}
        users_voted = set()
        voted = 0
        skip = 0

        for (name, dist) in neighbors:
            (userid,tags) = self.textstore.get(name, (None, None))
            if tags is None or userid in users_voted:
                skip += 1
                continue
            users_voted.add(userid)
            tagset = set(tags.split())
            for tag in tagset:
                tag2vote[tag] = tag2vote.get(tag,0) + 1
            voted += 1
       
        #assert(voted >= self.k), 'too many skips (%d) in %d neighbors' % (skip, len(neighbors))
        return sorted(tag2vote.items(), key=operator.itemgetter(1), reverse=True)

if __name__ == '__main__':
    TagVoteTagger('geonustraindev', '/Users/xirong/VisualSearch')

