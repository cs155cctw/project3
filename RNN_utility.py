import os
import re
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

def get_syllabus(line, idx, syll_map):
    syll = '0'
    word = re.sub(r'[^\w]', '', line[idx]).lower()
    if word not in syll_map:
        return syll

    if len(syll_map[word]) == 1:
        syll = syll_map[word][0]
    else:
        if syll_map[word][0][0] == 'E':
            if idx == len(line)-1:
                syll = syll_map[word][0]
            else:
                syll = syll_map[word][1]
        elif syll_map[word][1][0] == 'E':
            if idx == len(line)-1:
                syll = syll_map[word][1]
            else:
                syll = syll_map[word][0]
        else:
            sum_syll = 0
            for idx2 in range(len(line)):
                word2 = re.sub(r'[^\w]', '', line[idx2]).lower()
                if word2 not in syll_map:
                    continue
                if len(syll_map[word2]) == 1:
                    sum_syll += int(syll_map[word][0])
            syll = syll_map[word][0]
            if sum_syll+int(syll_map[word][1]) == 10:
                syll = syll_map[word][1]
    return syll


def poem_to_id(filename):
    '''
    output: an array of sequence, each sequence is an array of encoded word for a poem, a stanza, or a line
    '''

    # Convert text to dataset.
    text = open(os.path.join(os.getcwd(), filename)).read()
    lines = [line.split() for line in text.split('\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_map = {}

#    # read syllables map
#    syll_map = {}
#    text_syll = open(os.path.join(os.getcwd(), 'data/Syllable_dictionary.txt')).read()
#    lines_syll = [line.split() for line in text_syll.split('\n') if line.split()]

#    for line in lines_syll:
#        word = line[0]
#        word = re.sub(r'[^\w]', '', word).lower()
#        #print(word)
#        sylls = []
#        for idx in range(1, len(line)):
#            sylls.append(line[idx])
#        syll_map[word] = sylls

    # read state map
#    state_map = {}
#    text_state = open(os.path.join(os.getcwd(), 'data/cmudict.dict')).read()
#    lines_state = [line.split() for line in text_state.split('\n') if line.split()]

#    debug_line = 0
#    for line in lines_state:
#        word = line[0]
#        word = re.sub(r'[^\w]', '', word).lower()
#        sylls_all = ['0', '1', '01', '10', '001', '010', '100', '0010',
#                     '0100', '1000']
#        #0, 1,  2,  3,   4,   5,   6,    7,    8,    9,   10(NA),  11(punction)
#        sylls = ''
#        for idx in range(1, len(line)):
#            if line[idx][-1] == '1':
#                sylls = sylls+'1'
#            if line[idx][-1] == '0' or line[idx][-1] == '2':
#                sylls = sylls+'0'
#
#        state = 10
#
#        for idx in range(len(sylls_all)):
#            if sylls == sylls_all[idx]:
#                state = idx
#        state_map[word] = state
#        #print(state)
#        debug_line += 1

#    print(state_map['from'])
    for idx in range(len(lines)):
        line = lines[idx]
        # ignore poem number
        if idx % 15 < 1:
            continue
#        word_index_in_line = 0
        for word in line:
            if (word == ',') or (word == '.') or (word == '?') or (word == '!') or (word == ';') or (word == ':'):
                word = word#+'_0'
            else:
                word = re.sub(r'[^\w]', '', word).lower()
                if word == '':
                    continue
#                ####add syllables as suffix of the word####
#                ### first test, no syllables added ###
#                word = word+'_'+get_syllabus(line, word_index_in_line, syll_map)
#                ########
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
            # Add the encoded word.
            obs.append(obs_map[word])
#            word_index_in_line += 1

    # map all poems into integer index in order

    return obs, obs_map

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
    #y = np.zeros((batch_size, seq_size))
    #y = np.zeros((batch_size, seq_size, dict_len))
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
        #y[idx, :] = corpus[idx_start+1:idx_end+1]
        #tmp = corpus[idx_start+1:idx_end+1]
        #y[idx,:,:] = to_categorical(tmp, num_classes=dict_len)
        idx_start += skip_step

    return x, y

def sortSecond(val):
    return val[1]

def id_to_sentence(id, dict):
    '''
    draw word from dictionary according to word idx
    id_list = a list of word id in dictionary
    dict = dictionary built from corpus
    '''
    dict_list = list(dict.items())
    dict_list.sort(key = sortSecond)
    sentence = ''
    #for id in id_list:
        #id = int(id)
    sentence += dict_list[id][0] + ' '
    return sentence


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
        test = model.predict(seed_sen)
        prediction = model.predict(seed_sen)[0]
        id = sample(prediction, temperature)
        #id = id_list[:,seq_size-1]
        #id_list = np.argmax(prediction[:, seq_size-1, :])
        pseudo_poem += id_to_sentence(id, dict) + ' '
        next_sen = np.zeros((1, seq_size, dict_len))
        next_sen[0,:seq_size-1,:] = seed_sen[0, 1:, :]
        next_sen[0,seq_size-1,:] = to_categorical(id, num_classes=dict_len)
        seed_sen = next_sen
    return pseudo_poem
