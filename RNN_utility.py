import os
import re
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

def poem_to_id(filename):
    '''
    output: an array of sequence, each sequence is an array of encoded word for a poem, a stanza, or a line
    '''

    # Convert text to dataset.
    text = open(os.path.join(os.getcwd(), filename)).read()
    lines = [line.split() for line in text.split('\n') if line.split()]

    char_counter = 0
    char_idx = []
    char_map = {}
    raw_corpus = ''

    for idx in range(len(lines)):
        line = lines[idx]
        for word_idx in range(len(line)):
            word = line[word_idx]
            for char in word:
                if (char == ',') or (char == '.') or (char == '?') or (char == '!') or (char == ';') or (char == ':') or (char == '-') or ( char == "'" ):
                    raw_corpus+= char
                else:
                    char = re.sub(r'[^\w]', '', char).lower()
                    if char == '':
                        continue
                    raw_corpus+= char
            if word_idx == len(line) - 1:
                raw_corpus += '\n'
            else:
                raw_corpus += ' '

    for char in raw_corpus:
        if char not in char_map:
            # Add unique words to the observations map.
            char_map[char] = char_counter
            char_counter += 1
            # Add the encoded word.
        char_idx.append(char_map[char])

    return char_idx, char_map

def sentence_to_id(seed, dict):
    seed_id = []
    dict_len = len(dict)
    for char in seed:
        id = dict[char]
        tmp = to_categorical(id, num_classes = dict_len)
        seed_id.append(tmp)
    return seed_id

def RNN_data_generator(corpus, seq_size, batch_size, dict_len, skip_step):
    '''
    generate input x and y from the whole corpus
    poem = whole corpus mapped into integer index in order
    seq_size = number of consecutive characters from corpus that forms a sequence
    batch_size = number of sequences to train over
    dict_len = length of dictionary built from the corups
    skip_step = gap between consecutive sequences
    '''

    x = np.zeros((batch_size, seq_size, dict_len))
    y = np.zeros((batch_size, dict_len))
    corpus_len = len(corpus)
    idx_start = 0
    idx_end = 0
    for idx in range(batch_size):
        idx_start = idx_start % corpus_len
        idx_end = idx_start + seq_size
        if idx_end+1 >= corpus_len:
            print('redundant sequences, choose smaller batch_size')
            idx_start = int(round(skip_step/2.0))
            idx_end = idx_start + seq_size
        tmp_x = corpus[idx_start:idx_end]
        x[idx, :, :] = to_categorical(tmp_x, num_classes=dict_len)
        tmp_y = corpus[idx_end+1]
        y[idx, :] = to_categorical(tmp_y, num_classes=dict_len)
        idx_start += skip_step

    return x, y

def sortSecond(val):
    return val[1]

def id_to_char(id, dict):
    '''
    draw word from dictionary according to word idx
    id_list = a list of word id in dictionary
    dict = dictionary built from corpus
    '''
    dict_list = list(dict.items())
    dict_list.sort(key = sortSecond)
    char = dict_list[id][0]
    return char


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def poem_composer(model, seed_sen, poem_len, seq_size, dict, temperature):
    '''
    generate poems from RNN model
    seed_sen = seed sentence fed into the model
    poem_len = length of poem to write
    '''
    pseudo_poem = ''
    dict_len = len(dict)
    for idx in range(poem_len):
        for char_idx in range(seq_size):
            test = model.predict(seed_sen)
            prediction = model.predict(seed_sen)[0]
            id = sample(prediction, temperature)
            #id = id_list[:,seq_size-1]
            pseudo_poem += id_to_char(id, dict)
            next_sen = np.zeros((1, seq_size, dict_len))
            next_sen[0,:seq_size-1,:] = seed_sen[0, 1:, :]
            next_sen[0,seq_size-1,:] = to_categorical(id, num_classes=dict_len)
            seed_sen = next_sen
        pseudo_poem += ' \n '
    return pseudo_poem
