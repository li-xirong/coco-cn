def decode_sequence(vocab, seq):
    # print(seq)
    # print(vocab.word2idx['a'])
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            index = seq[i, j].item()
            txt = txt + ' '
            if vocab.idx2word[index] == '<end>' or vocab.idx2word[index] == '<pad>':
                break
            txt = txt + vocab.idx2word[index]
        out.append(txt.strip())
    return out
