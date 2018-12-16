import nltk
import pickle
import argparse
from collections import Counter
from tqdm import tqdm 
import json

from basic.common import ROOT_PATH as rootpath
import basic.path_util as path_util
import os

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(dataset_json_path, threshold):
    """Build a simple vocabulary wrapper."""
    with open(dataset_json_path) as f:
        dataset = json.loads(f.readline())
    images = dataset['images']
    counter = Counter()
    for image in tqdm(images):
        for sent in image['sentences']:
            counter.update(sent['tokens'])

    # If the word frequency is less than 'threshold', then the word is
    # discarded.
    words = [word for word, cnt in counter.items() if cnt > threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab


def main(args):
 
    collection = args.collection
    rootpath = args.rootpath
    input_json = path_util.get_input_json(collection, 'train', rootpath=rootpath)
    vocab = build_vocab(input_json, args.threshold)
    vocab_path = path_util.get_vocab(collection, rootpath=rootpath)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" % len(vocab))
    print("Saved the vocabulary wrapper to '%s'" % vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', default=rootpath, help='rootpath of the data')
    parser.add_argument('--collection', required=True, help='collection')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
