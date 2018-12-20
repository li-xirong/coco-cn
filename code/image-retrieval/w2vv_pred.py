import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json

class W2VV_pred(object):
    def __init__(self, model_path, weight_path, text2vec, sent_maxlen, embed_size, bow2vec, w2v2vec):

        self.w2vv_model = model_from_json(open(model_path).read())
        self.w2vv_model.load_weights(weight_path)
        self.w2vv_model.compile(loss='mse', optimizer='rmsprop')
        self.text2vec = text2vec
        self.bow2vec = bow2vec
        self.w2v2vec = w2v2vec
        self.sent_maxlen = sent_maxlen

    # predict a visual feature vector for a given text input
    def predict_one(self, text_input):
        text_embedding = self.text2vec.embedding(text_input)

        bow_embedding = self.bow2vec.embedding(text_input)
        w2v_embedding = self.w2v2vec.embedding(text_input)
        # fasttext_embedding = self.fasttext2vec.embedding(text_input)
        # text string not in the word2vec vocabulary
        if text_embedding is None:
            text_embedding = np.zeros(self.text2vec.ndims)
        if bow_embedding is None:
            bow_embedding = np.zeros(self.bow2vec.ndims)
        if w2v_embedding is None:
            w2v_embedding = np.zeros(self.w2v2vec.ndims)
        #if text_embedding is None or bow_embedding is None or w2v_embedding is None:
        #    return None
        text_embedding = pad_sequences([list(text_embedding)], maxlen=self.sent_maxlen)[0]
        pred_visual_feat_vec = self.w2vv_model.predict([np.array([text_embedding]), np.array([list(bow_embedding) + list(w2v_embedding)])])[0]
        return pred_visual_feat_vec.tolist()

    # predict visual feature vectors for a given list of text input
    # return:
    # success_text_input_index: the index of sucess input text
    # pred_visual_feat_vecs: the predicted visual feature vectors
    def predict_batch(self, text_input_batch):
        success_text_input_index = []
        text_embedding_batch = []
        bow_embedding_batch = []
        for index, text_input in enumerate(text_input_batch):
            text_embedding = self.text2vec.embedding(text_input)
            bow_embedding = self.bow2vec.embedding(text_input)
            w2v_embedding = self.w2v2vec.embedding(text_input)
            if text_embedding is not None and bow_embedding is not None and w2v_embedding is not None:
                success_text_input_index.append(index)
                text_embedding_batch.append(text_embedding)
                bow_embedding_batch.append(list(bow_embedding) + list(w2v_embedding))

        pred_visual_feat_vecs = self.w2vv_model.predict([np.array(text_embedding_batch), np.array(bow_embedding_batch)])
        
        return success_text_input_index, pred_visual_feat_vecs.tolist()
