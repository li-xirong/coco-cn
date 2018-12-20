import random
import numpy as np

from keras.preprocessing.sequence import pad_sequences

# dataset of image-sentence pairs 
class BaseData(object):
    def __init__(self, data_file, batch_size, qry2vec, bow2vec, w2v2vec, img_feats, flag_maxlen=False, maxlen=32):
        # image-query pairs
        all_data = open(data_file).readlines()
        random.shuffle(all_data)

        img_list = []
        query_list = []
        for line in all_data:
            sid, sent = line.strip().split(' ', 1)
            if qry2vec.mapping(sent) is None or bow2vec.mapping(sent) is None or w2v2vec.mapping(sent) is None:
                print sent
                continue
            query_list.append(sent)
            img = sid.strip().split('#')[0]
            img_list.append(img)
 
        self.imgs = img_list
        self.querys = query_list
        assert len(img_list) == len(query_list)
 
        self.datasize = len(img_list)
        self.qry2vec = qry2vec
        self.bow2vec = bow2vec
        self.w2v2vec = w2v2vec
        self.img_feats = img_feats
 
        self.batch_size = batch_size
        self.max_batch_size = int( np.ceil(1.0 * len(img_list) / batch_size))

        self.flag_maxlen = flag_maxlen
        self.maxlen = maxlen


class PairDataSet_Sent(BaseData):
    def getBatchData(self):
        counter = 0
        while 1:
            query_list = self.querys[counter*self.batch_size: (counter+1)*self.batch_size]
            # print query_list
            img_list = self.imgs[counter*self.batch_size: (counter+1)*self.batch_size]
     
            X = []
            X_1 = []
            X_2 = []
            for query in query_list:
                X.append(self.qry2vec.mapping(query))
                X_1.append(list(self.bow2vec.mapping(query)) + list(self.w2v2vec.mapping(query)))
                
            Y = []
            renamed, feats = self.img_feats.read(img_list)
            for img in img_list:
               Y.append(feats[renamed.index(img)])

            if self.flag_maxlen:
                X = pad_sequences(X, maxlen=self.maxlen, truncating='post')

            # X=query Y=image
            yield ([np.array(X), np.array(X_1)], np.array(Y))
            counter += 1
            counter = counter % self.max_batch_size

            # reshuffle the training data after each epoch
            if counter == 0:
                query_img = zip(self.querys, self.imgs)
                random.shuffle(query_img)
                self.querys = [x[0] for x in query_img]
                self.imgs = [x[1] for x in query_img]


class PairDataSet_CTL(BaseData):
    def getBatchData(self):
        counter = 0
        while 1:
            query_list = self.querys[counter*self.batch_size: (counter+1)*self.batch_size]
            # print query_list
            img_list = self.imgs[counter*self.batch_size: (counter+1)*self.batch_size]

            constr_img_list = [img for img in self.imgs if img not in img_list]
     
            X = []
            X_1 = []
            X_2 = []
            for query in query_list:
                X.append(self.qry2vec.mapping(query))
                X.append(self.qry2vec.mapping(query))
                X_1.append(list(self.bow2vec.mapping(query)) + list(self.w2v2vec.mapping(query)))
                X_1.append(list(self.bow2vec.mapping(query)) + list(self.w2v2vec.mapping(query)))
                
            Y = []
            flag_Y = []
            renamed, feats = self.img_feats.read(img_list)
            for img in img_list:
               Y.append(feats[renamed.index(img)])
               flag_Y.append([1])
               Y.append(self.img_feats.read_one(random.choice(constr_img_list)))
               flag_Y.append([0])

            if self.flag_maxlen:
                X = pad_sequences(X, maxlen=self.maxlen, truncating='post')

            # X=query Y=image
            yield ([np.array(X), np.array(X_1)], np.concatenate((np.array(flag_Y), np.array(Y)), axis=1))
            counter += 1
            counter = counter % self.max_batch_size

            # reshuffle the training data after each epoch
            if counter == 0:
                query_img = zip(self.querys, self.imgs)
                random.shuffle(query_img)
                self.querys = [x[0] for x in query_img]
                self.imgs = [x[1] for x in query_img]


NAME_TO_DATASET = { 
                   'sent_mse': PairDataSet_Sent,
                   'sent_mrl': PairDataSet_Sent,
                   'sent_ctl': PairDataSet_CTL
                   }

def get_dataset(name):
    return NAME_TO_DATASET[name]
