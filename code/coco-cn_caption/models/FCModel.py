# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from models.CaptionModel import CaptionModel


class LSTMCore(nn.Module):
    def __init__(self, input_encoding_size, rnn_size, drop_prob_lm):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.drop_prob_lm = drop_prob_lm

        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, state):

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = torch.max(
            all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
            all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        next_h = self.dropout(next_h)

        output = next_h
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


class FCModel(CaptionModel):
    """docstring for FCModel"""

    def __init__(self, opt):
        super(FCModel, self).__init__()
        self.fc_feat_size = opt.fc_feat_size
        self.lstm_input_size = opt.lstm_input_size
        self.lstm_hidden_size = opt.lstm_hidden_size
        self.lstm_num_layers = opt.lstm_num_layers
        self.lstm_drop_prob = opt.lstm_drop_prob
        self.vocab_size = opt.vocab_size
        self.word_embed_size = opt.word_embed_size

        self.seq_length = opt.seq_length
        self.feedback_prob = opt.feedback_prob_start

        self.img_embed = nn.Linear(self.fc_feat_size, self.lstm_input_size)
        self.word_embed = nn.Embedding(self.vocab_size, self.word_embed_size)

        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers, dropout=self.lstm_drop_prob)
        self.lstmcore = LSTMCore(
            self.lstm_input_size, self.lstm_hidden_size, self.lstm_drop_prob)
        self.lstmcell = nn.LSTMCell(
            input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size)

        self.logit = nn.Linear(self.lstm_hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(self.lstm_drop_prob)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_lstm_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.lstm_num_layers, batch_size, self.lstm_hidden_size).zero_()),
                Variable(weight.new(self.lstm_num_layers, batch_size, self.lstm_hidden_size).zero_()))

    def init_lstmcell_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(batch_size, self.lstm_hidden_size).zero_()),
                Variable(weight.new(batch_size, self.lstm_hidden_size).zero_()))

                
    # use lstmcore
    def forward(self, fc_feats, captions, lengths):
        batch_size = fc_feats.size(0)
        state = self.init_lstm_hidden(batch_size)
        outputs = []
        logoutputs = []
        for t in range(captions.size(1) + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1:
                    wt = fc_feats.data.new(batch_size).long().fill_(1)
                    wt = Variable(wt, requires_grad=False)
                elif self.training and t > 1 and self.feedback_prob > 0.0:
                    sample_prob = torch.FloatTensor(
                        batch_size).cuda().uniform_(0, 1)
                    sample_mask = sample_prob < self.feedback_prob
                    # print("I'm in !!!!!")
                    if sample_mask.sum() == 0:
                        wt = captions[:, t - 2].clone()
                    else:
                        select_index = sample_mask.nonzero().view(-1)
                        # print(select_index)
                        prob_prev = torch.exp(logprobs)
                        wt = captions[:, t - 2].data.clone()
                        wt_sample = torch.multinomial(prob_prev, 1)
                        # print(wt_sample)
                        select_sample = wt_sample.index_select(
                            0, Variable(select_index, requires_grad=False))
                        wt.index_copy_(0, select_index, select_sample.view(-1))#data)
                        wt = Variable(wt, requires_grad=False)
                else:
                    wt = captions[:, t - 2].clone()

                xt = self.word_embed(wt)
            output, state = self.lstmcore(xt, state)
            output = self.logit(output)
            logprobs = F.log_softmax(output)
            if t > 0:
                outputs.append(output)
                logoutputs.append(logprobs)

        return torch.cat([_.unsqueeze(1) for _ in outputs[:-1]], 1).contiguous(), torch.cat([_.unsqueeze(1) for _ in logoutputs[:-1]], 1).contiguous()

    def get_logprobs_state(self, wt, state):
        # 'it' is Variable contraining a word index
        xt = self.word_embed(wt)

        output, state = self.lstmcore(xt, state)
        logprobs = F.log_softmax(self.logit(output))

        return logprobs, state

    def sample_beam(self, fc_feats, opt={}):
        beam_size = opt.get('beam_size', 5)
        batch_size = fc_feats.size(0)

        seq = torch.LongTensor(self.seq_length, beam_size, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, beam_size, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_lstm_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(
                        fc_feats[k:k + 1]).expand(beam_size, self.lstm_input_size)
                elif t == 1:  # input <bos>
                    wt = fc_feats.data.new(beam_size).long().fill_(1)
                    xt = self.word_embed(Variable(wt, requires_grad=False))

                output, state = self.lstmcore(xt, state)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)

            # save all the beams
            for i in range(beam_size):
                seq[:, i, k] = self.done_beams[k][i]['seq']
                seqLogprobs[:, i, k] = self.done_beams[k][i]['logps']
            
        return seq.transpose(0, 2), seqLogprobs.transpose(0, 2)

    def sample_beam_with_first_result(self, fc_feats, opt={}):
        beam_seq, beam_seqLogprobs = self.sample_beam(fc_feats, opt=opt)
        # the first beam has highest cumulative score
        seq = beam_seq[:,0,:]
        seqLogprobs = beam_seqLogprobs[:,0,:]
        return seq, seqLogprobs


    # use lstmcore
    def sample(self, fc_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        if beam_size > 1:
            #return self.sample_beam(fc_feats, opt)
            return self.sample_beam_with_first_result(fc_feats, opt)
        batch_size = fc_feats.size(0)
        state = self.init_lstm_hidden(batch_size)
        # seq worlds index
        seq = []
        # seqLogprabs序列中对应元素对数正太分布的概率
        seqLogprobs = []
        for t in range(self.seq_length + 2):
            # print('*'*10, t)
            # --xt为LSTM网络中输入的向量x，it是词向量，sampleLogprobs为sample的对数正态分布的概率
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1:  # input <bos>
                    # -- feed in the start tokens
                    wt = fc_feats.data.new(batch_size).long().fill_(1)
                elif sample_max:
                    # -- use argmax "sampling"
                    # --取得predictions的最大的值，作为预测值
                    sampleLogprobs, wt = torch.max(logprobs.data, 1)
                    wt = wt.view(-1).long()
                else:
                    prob_prev = torch.exp(logprobs.data).cpu()
                    # sample
                    wt = torch.multinomial(prob_prev, 1).cuda()
                    # gather the logprobs at sampled positions
                    sampleLogprobs = logprobs.gather(
                        1, Variable(wt, requires_grad=False))
                    # and flatten indices for downstream processing
                    wt = wt.view(-1).long()

                xt = self.word_embed(Variable(wt, requires_grad=False))
            if t > 1:
                if t == 2:
                    unfinished = wt > 0
                else:
                    unfinished = unfinished * (wt > 0)
                if unfinished.sum() == 0:
                    break
                wt = wt * unfinished.type_as(wt)
                seq.append(wt)  # seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            # print(xt)
            output, state = self.lstmcore(xt, state)
            logit_output = self.logit(output)
            logprobs = F.log_softmax(logit_output)

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
