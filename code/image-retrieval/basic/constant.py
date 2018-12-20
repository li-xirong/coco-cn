import os

ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')

DEFAULT_TEXT_STYLE = 'gru@bow_filterstop@word2vec_filterstop'
DEFAULT_IMG_FEATURE = 'pyresnet152-pool5os'

DEFAULT_BOW_VOCAB = 'word_vocab_5.txt'
DEFAULT_RNN_VOCAB = 'word_vocab_5.txt'

DEFAULT_CORPUS = 'flickr'
DEFAULT_WORD2VEC = 'vec500flickr30m'

FULL_COLLECTION = 'mscoco2014dev'

