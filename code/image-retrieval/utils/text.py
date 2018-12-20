# -*- coding: utf-8 -*-

import re
import sys
import numpy as np
from simpleknn.bigfile import BigFile
from gensim.models import word2vec as w2v


ENGLISH_STOP_WORDS = map(str.strip, open('en_stopwords.txt'))
CHINESE_STOP_WORDS = map(str.strip, open('zh_stopwords.txt').readlines())

if 3 == sys.version_info[0]:
    CHN_DEL_SET = '， 。 、 ！ 《 》 “ ” ； ？ ‘ ’ '.split()
else:
    CHN_DEL_SET = [x.decode('utf-8') for x in '， 。 、 ！ 《 》 “ ” ； ？ ‘ ’ '.split()]

class TextTool:
    @staticmethod
    def tokenize(input_str, language='en'):
        if 'en' == language: # English
            # delete non-ascii chars
            #sent = input_str.decode('utf-8').encode('ascii', 'ignore')
            sent = input_str
            sent = sent.replace('\r',' ')
            sent = re.sub(r"[^A-Za-z0-9]", " ", sent).strip().lower()
            tokens = sent.split()
        else: # Chinese  
        # sent = input_str #string.decode('utf-8')
            sent = input_str.decode('utf-8')
            for elem in CHN_DEL_SET:
                sent = sent.replace(elem,'')
            sent = sent.encode('utf-8')
            sent = re.sub("[A-Za-z]", "", sent)
            tokens = [x for x in sent.split()] 

        return tokens

# language in [en, zh]
def clean_str(string, language):
    return TextTool.tokenize(string, language)


def clean_str_filter_stop(string, language):
    cleaned_string = TextTool.tokenize(string, language)
    # remove stop words
    if 'en' == language:
        return [word for word in cleaned_string if word not in ENGLISH_STOP_WORDS]
    else:
        return [word for word in cleaned_string if word not in CHINESE_STOP_WORDS]


def get_cn_we_parameter(vocabulary, word2vec_file):
    print 'getting inital word embedding ...'
    #w2v_reader = BigFile(word2vec_file)
    w2v_model = w2v.Word2Vec.load(word2vec_file)
    ndims = 500
    fail_counter = 0
    we = []
    # Reserve 0 for masking via pad_sequences
    we.append([0]*ndims)
    for word in vocabulary:
        word = word.strip()
        try:
            vec = w2v_model.wv[unicode(word.decode('utf-8'))]
            we.append(vec)
        except Exception, e:
            vec = np.random.uniform(-1,1, ndims)
            we.append(vec)
            fail_counter +=1
    print "%d words out of %d words cannot find pre-trained word2vec vector" % (fail_counter, len(vocabulary))
    return np.array(we)

def get_en_we_parameter(vocabulary, word2vec_file):
    print 'getting inital word embedding ...'
    w2v_reader = BigFile(word2vec_file)
    ndims = w2v_reader.ndims
    fail_counter = 0
    we = []
    # Reserve 0 for masking via pad_sequences
    we.append([0]*ndims)
    for word in vocabulary:
        word = word.strip()
        try:
            vec = w2v_reader.read_one(word)
            # print vec
            we.append(vec)
        except Exception, e:
            vec = np.random.uniform(-1,1,ndims)
            we.append(vec)
            fail_counter +=1
    print "%d words out of %d words cannot find pre-trained word2vec vector" % (fail_counter, len(vocabulary))
    return np.array(we)

def get_we_parameter(vocabulary, word2vec_file, style='en'):
    if style=='en':
        return get_en_we_parameter(vocabulary, word2vec_file)
    else:
        return get_cn_we_parameter(vocabulary, word2vec_file)


if __name__ == '__main__':
    test_strs = '''a Dog is running
The dog runs
dogs-x runs'''.split('\n')

    for t in test_strs:
        print t, '->', clean_str(t, 'en'), '->', clean_str_filter_stop(t, 'en')

