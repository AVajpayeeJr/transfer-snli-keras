from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import os


__author__ = 'avijitv'


def get_nli_dataset(path):
    dataset = {'train': {}, 'dev': {}, 'test': {}}
    label_codes = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    for data_type in ['train', 'dev', 'test']:
        s1_path = os.path.join(path, 's1.' + data_type)
        s2_path = os.path.join(path, 's2.' + data_type)
        target_path = os.path.join(path, 'labels.' + data_type)
        dataset[data_type] = {}
        dataset[data_type]['s1'] = [text_to_word_sequence(line.rstrip()) for line in
                                    open(s1_path, 'r')]
        dataset[data_type]['s2'] = [text_to_word_sequence(line.rstrip()) for line in
                                    open(s2_path, 'r')]
        dataset[data_type]['target'] = np.array([label_codes[line.rstrip('\n')]
                                                 for line in open(target_path, 'r')])

        print('{0} : Found {1} pairs.'.format(data_type.upper(),
                                              len(dataset[data_type]['s1']),
                                              data_type))

    return dataset


def get_vocab(sentences):
    vocab = {'<pad>': 0, '<eos>': 1, '<unk>': 2}
    word_cnt = 3
    for s in sentences:
        for word in s:
            if word not in vocab:
                vocab[word] = word_cnt
                word_cnt += 1
    return vocab


def get_embeddings_matrix(embeddings_path, vocab, embedding_dim=300):
    word_embedding_matrix = np.zeros((len(vocab), embedding_dim))

    found = set([])
    with open(embeddings_path, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line:
                t = line.split(' ')
                word = t[0]
                if word in vocab:
                    idx = vocab[word]
                    vec = np.array([float(j) for j in t[1:]])
                    word_embedding_matrix[idx, :] = vec
                    found.add(word)

    not_found = set(vocab.keys()).difference(found)
    print('Found embeddings: {0} / {1} ({2} %)'.format(len(found),
                                                       len(vocab),
                                                       round(len(found) / len(vocab), 3)))
    print('Not found embeddings: {0} / {1} ({2} %)'.format(len(not_found),
                                                           len(vocab),
                                                           round(len(not_found) / len(vocab), 3)))
    for word in not_found:
        idx = vocab[word]
        unk_idx = vocab['<unk>']
        word_embedding_matrix[idx, :] = word_embedding_matrix[unk_idx, :]

    word_embedding_matrix[vocab['<pad>'], :] = np.array([0.0] * embedding_dim)

    return word_embedding_matrix


def get_sequences(sentences, vocab, max_seq_len):
    seq = [[vocab[j] if j in vocab else vocab['<unk>'] for j in s] for s in sentences]
    return pad_sequences(maxlen=max_seq_len, sequences=seq, padding="post", value=0)
