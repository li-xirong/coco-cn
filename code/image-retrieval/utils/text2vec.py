import os
import numpy as np
from gensim.models import word2vec as w2v

from simpleknn.bigfile import BigFile
from basic.common import printStatus
from basic.constant import ROOT_PATH as rootpath

from text import clean_str, clean_str_filter_stop

INFO = __file__


class Text2Vec:

    def __init__(self, datafile, ndims=0, language='en', L1_normalize=0, L2_normalize=0, k=5):
        printStatus(INFO + '.' + self.__class__.__name__, 'initializing ...')
        self.datafile = datafile
        self.k = k
        self.ndims = ndims
        self.L1_normalize = L1_normalize
        self.L2_normalize = L2_normalize
        self.language = language

        assert type(L1_normalize) == int
        assert type(L2_normalize) == int
        assert (L1_normalize + L2_normalize) <= 1

    def embedding(self, query):
        vec = self.mapping(query)
        if vec is not None:
            vec = np.array(vec)
        return vec

    def do_L1_norm(self, vec):
        L1_norm = np.linalg.norm(vec, 1)
        return 1.0 * np.array(vec) / L1_norm

    def do_L2_norm(self, vec):
        L2_norm = np.linalg.norm(vec, 2)
        return 1.0 * np.array(vec) / L2_norm


# word2vec + average pooling
class AveWord2Vec(Text2Vec):

    # datafile: the path of pre-trained word2vec data
    def __init__(self, datafile, ndims=0, language='en', L1_normalize=0, L2_normalize=0):
        Text2Vec.__init__(self, datafile, ndims, language, L1_normalize, L2_normalize)
        self.word2vec = BigFile(datafile) if language=='en' else w2v.Word2Vec.load(datafile)
        if ndims != 0:
            if 'en'==language:
                assert self.word2vec.ndims == self.ndims, "feat dimension is not match %d != %d" % (self.word2vec.ndims, self.ndims)
            else:
                print 'ndims #', ndims
        else:
            self.ndims = self.word2vec.ndims if 'en'==language else 500


    def preprocess(self, query, clear):
        if clear:
            words = clean_str(query, self.language)
        else:
            words = query.strip().split()
        return words

    def mapping(self, query, clear = True):
        words = self.preprocess(query, clear)

        #print query, '->', words
        if 'en'==self.language:
            renamed, vectors = self.word2vec.read(words)
            renamed2vec = dict(zip(renamed, vectors))

            if len(renamed) != len(words):
                vectors = []
                for word in words:
                    if word in renamed2vec:
                        vectors.append(renamed2vec[word])
        else:
            vectors = []
            for word in words:
                try:
                    vec = self.word2vec.wv[unicode(word.decode('utf-8'))]
                    if vec is not None:
                        vectors.append(vec)
                except KeyError:
                    #print word, 'not in w2v vocabulary'
                    pass

        if len(vectors)>0:
            vec = np.array(vectors).mean(axis=0)

            if self.L1_normalize:
                return self.do_L1_norm(vec)
            if self.L2_normalize:
                return self.do_L2_norm(vec)

            return vec
        else:
            return None

# word2vec + average pooling + fliter stop words
class AveWord2VecFilterStop(AveWord2Vec):

    def preprocess(self, query, clear):
        if clear:
            words = clean_str_filter_stop(query, self.language)
        else:
            words = query.strip().split()
        return words       


# Bag-of-words
class BoW2Vec(Text2Vec):

    # datafile: the path of bag-of-words vocabulary file
    def __init__(self, datafile, ndims=0, language='en', L1_normalize=0, L2_normalize=0):
        Text2Vec.__init__(self, datafile, ndims, language, L1_normalize, L2_normalize)
        word_vob = map(str.strip, open(datafile).readlines())
        self.word2index = dict(zip(word_vob, range(len(word_vob))))

        if ndims != 0:
            assert len(word_vob) == self.ndims, "feat dimension is not match %d != %d" % (len(word_vob), self.ndims)
        else:
            self.ndims = len(word_vob)
        printStatus(INFO + '.' + self.__class__.__name__, "%d words" % self.ndims)

    
    def preprocess(self, query):
        return clean_str(query, self.language)

    def mapping(self, query):
        words = self.preprocess(query)

        vec = [0.0]*self.ndims
        for word in words:
            if word in self.word2index:
                vec[self.word2index[word]] += 1
            # else:
            #     print word

        if self.L1_normalize:
            return self.do_L1_norm(vec)
        if self.L2_normalize:
            return self.do_L2_norm(vec)

        if sum(vec) > 0:
            return vec
        else:
            return None

# Bag-of-words
class BoW2VecSoft(Text2Vec):

    # datafile: the path of bag-of-words vocabulary file
    def __init__(self, datafile, k=5, ndims=0, language='en', L1_normalize=0, L2_normalize=0):
        Text2Vec.__init__(self, datafile, k, ndims, language, L1_normalize, L2_normalize)
        self.words_simi = {}
        word_vob = map(str.strip, open(datafile).readlines())
        self.word2index = dict(zip(word_vob, range(len(word_vob))))
        self.soft_file = os.path.join(datafile.rsplit('/', 1)[0], 'word_vocab_soft_5.txt')
        for line in open(self.soft_file).readlines():
            word, s_sim = line.strip().split(' ', 1)
            w_s = s_sim.split(' ')
            assert len(w_s) % 2 == 0
            self.words_simi[word] = [(w_s[i], float(w_s[i+1])) for i in range(0, len(w_s), 2)]

        if ndims != 0:
            assert len(word_vob) == self.ndims, "feat dimension is not match %d != %d" % (len(word_vob), self.ndims)
        else:
            self.ndims = len(word_vob)
        printStatus(INFO + '.' + self.__class__.__name__, "%d words" % self.ndims)

    
    def preprocess(self, query):
        return clean_str(query, self.language)

    def mapping(self, query):
        words = self.preprocess(query)
        vec = [0.0]*self.ndims
        for word in words:
            if word in self.word2index:
                vec[self.word2index[word]] += 1

            if word in self.words_simi:
                word_simi = self.words_simi[word]
                assert word_simi[0][0] == word and word_simi[0][1] == 1.0
                for i in range(1, self.k+1):
                    vec[self.word2index[word_simi[i][0]]] += word_simi[i][1]

        if self.L1_normalize:
            return self.do_L1_norm(vec)
        if self.L2_normalize:
            return self.do_L2_norm(vec)

        if sum(vec) > 0:
            return np.array(vec)
        else:
            return None

# Bag-of-words + fliter stop words
class BoW2VecFilterStop(BoW2Vec):

    def preprocess(self, query):
        return clean_str_filter_stop(query, self.language)


class BoW2VecFilterStopSoft(BoW2VecSoft):

    def preprocess(self, query):
        return clean_str_filter_stop(query, self.language)

# conver word to index 
class Index2Vec(Text2Vec):

    # datafile: the path of bag-of-words vocabulary file
    def __init__(self, datafile, ndims=0, language='en', L1_normalize=0, L2_normalize=0, maxlen=32, we_vocab=None):
        Text2Vec.__init__(self, datafile, ndims, language, L1_normalize, L2_normalize)

        if we_vocab is None:
            word_vob = map(str.strip, open(datafile).readlines())
        else:
            word_vob = we_vocab
        # Reserve 0 for masking via pad_sequences
        self.word2index = dict(zip(word_vob, range(1,len(word_vob)+1)))
        self.nvocab = len(word_vob) + 1
        self.maxlen = maxlen
        self.vocab = word_vob

    def preprocess(self, query):
        return clean_str(query, self.language)

    def mapping(self, query):
        words = self.preprocess(query)

        vec = [self.word2index[word] for word in words if word in self.word2index]        

        if len(vec) > 0:
            return vec[:self.maxlen]
        else:
            return None


# conver word to index + fliter stop words
class Index2VecFilterStop(Index2Vec):

    def preprocess(self, query):
        return clean_str_filter_stop(query, self.language)



NAME_TO_ENCODER = {'word2vec': AveWord2Vec, 'word2vec_filterstop': AveWord2VecFilterStop,
                   'bow': BoW2Vec, 'bow_filterstop': BoW2VecFilterStop, 'bow_filterstop_soft': BoW2VecFilterStopSoft,
                   'lstm': Index2Vec, 'bilstm': Index2Vec, 'gru': Index2Vec, 'bigru': Index2Vec,
                   'lstm_filterstop': Index2VecFilterStop}


def get_text_encoder(name):
    return NAME_TO_ENCODER[name]



if __name__ == '__main__':
    corpus = 'flickr'
    word2vec_model = 'vec500flickr30m'
    text_data_path = os.path.join(rootpath, "word2vec", corpus, word2vec_model)
    
    collection = 'msrvtt10k'

    encoder_w2vfs = get_text_encoder('word2vec_filterstop')(text_data_path, L1_normalize=1)

    query = "dog is running in boy"

    for encoder in [encoder_w2vfs]:
        feat = encoder.embedding(query)
        print(feat)
        print(sum(feat))
        #print len(feat), feat.min(), feat.max(), sum(feat)
