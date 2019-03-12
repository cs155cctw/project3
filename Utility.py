import os
import re
import numpy as np

def get_syllabus(line, idx, syll_map):
    syll = '0'
    word = re.sub(r'[^\w]', '', line[idx]).lower()
    word_last = re.sub(r'[^\w]', '', line[-1]).lower()
    haspunctuation = 0
    if word_last == '':
        haspunctuation = 1
    if word not in syll_map:
        return syll
    
    if len(syll_map[word]) == 1:
        syll = syll_map[word][0]
    else:
        if syll_map[word][0][0] == 'E':
            if idx == len(line)-1-haspunctuation:
                syll = syll_map[word][0]
            else:
                syll = syll_map[word][1]
        elif syll_map[word][1][0] == 'E':
            if idx == len(line)-1-haspunctuation:
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

    
def get_poem_sequence(filename, option):
    '''
    option: poem, line, stanza1, stanza2, stanza3, stanza4
    output: an array of sequence, each sequence is an array of encoded word for a poem, a stanza, or a line
    '''

    # Convert text to dataset.
    text = open(os.path.join(os.getcwd(), filename)).read()
    lines = [line.split() for line in text.split('\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_Y = []
    obs_map = {}
    obs_elem = []
    obs_Y_elem = []
    
    # read syllables map
    syll_map = {}
    text_syll = open(os.path.join(os.getcwd(), 'data/Syllable_dictionary.txt')).read()
    lines_syll = [line.split() for line in text_syll.split('\n') if line.split()]

    for line in lines_syll:
        word = line[0]
        word = re.sub(r'[^\w]', '', word).lower()
        #print(word)
        sylls = []
        for idx in range(1, len(line)):
            sylls.append(line[idx])
        syll_map[word] = sylls
        
    # read state map
    state_map = {}
    text_state = open(os.path.join(os.getcwd(), 'data/cmudict.dict')).read()
    lines_state = [line.split() for line in text_state.split('\n') if line.split()]
    
    debug_line = 0
    for line in lines_state:
        word = line[0]
        word = re.sub(r'[^\w]', '', word).lower()
        sylls_all = ['0', '1', '01', '10', '001', '010', '100', '0010', 
                     '0100', '1000']
        #0, 1,  2,  3,   4,   5,   6,    7,    8,    9,   10(NA),  11(punction)
        sylls = ''
        for idx in range(1, len(line)):
            if line[idx][-1] == '1':
                sylls = sylls+'1'
            if line[idx][-1] == '0' or line[idx][-1] == '2':
                sylls = sylls+'0'
        
        state = 10
        
        for idx in range(len(sylls_all)):
            if sylls == sylls_all[idx]:
                state = idx
        state_map[word] = state
        #print(state)
        debug_line += 1
    
    print(state_map['from'])
    for idx in range(len(lines)):
        line = lines[idx]
        # ignore poem number
        if idx % 15 < 1:
            continue
        if (option == 'line') or (option == 'poem') or (option == 'stanza1' and (idx % 15 >= 1 and idx % 15 < 5 )) or (option == 'stanza2' and (idx % 15 >= 5 and idx % 15 < 9 )) or (option == 'stanza3' and (idx % 15 >= 9 and idx % 15 < 13 )) or (option == 'stanza4' and (idx % 15 >= 13 and idx % 15 < 15 )):
            word_index_in_line = 0
            for word in line:
                if (word == ',') or (word == '.') or (word == '?') or (word == '!') or (word == ';') or (word == ':'):
                    word = word+'_0'
                    obs_Y_elem.append(11)
                else:
                    word = re.sub(r'[^\w]', '', word).lower()
                    if word == '':
                        continue
                    
                    if word in state_map:
                        obs_Y_elem.append(state_map[word])
                    else:
                        obs_Y_elem.append(10)
                    ####add syllables as suffix of the word####
                    word = word+'_'+get_syllabus(line, word_index_in_line, syll_map)
                    ########
                if word not in obs_map:
                    # Add unique words to the observations map.
                    obs_map[word] = obs_counter
                    obs_counter += 1
                # Add the encoded word.
                obs_elem.append(obs_map[word])
                word_index_in_line += 1

        if option == 'line':
            obs.append(obs_elem)
            obs_Y.append(obs_Y_elem)
            obs_elem = []
            obs_Y_elem = []

        if option == 'poem':
            if idx % 15 == 14:
                # Add the encoded sequence.
                obs.append(obs_elem)
                obs_Y.append(obs_Y_elem)
                # initialzie the sequence
                obs_elem = []
                obs_Y_elem = []

        if option == 'stanza1':
            if idx % 15 == 4:
                obs.append(obs_elem)
                obs_Y.append(obs_Y_elem)
                obs_elem = []
                obs_Y_elem = []
            if idx % 15 == 14:
                obs_elem = []
                obs_Y_elem = []

        if option == 'stanza2':
            if idx % 15 == 8:
                obs.append(obs_elem)
                obs_Y.append(obs_Y_elem)
                obs_elem = []
                obs_Y_elem = []
            if idx % 15 == 4:
                obs_elem = []
                obs_Y_elem = []

        if option == 'stanza3':
            if idx % 15 == 12:
                obs.append(obs_elem)
                obs_Y.append(obs_Y_elem)
                obs_elem = []
                obs_Y_elem = []
            if idx % 15 == 8:
                obs_elem = []
                obs_Y_elem = []

        if option == 'stanza4':
            if idx % 15 == 14:
                obs.append(obs_elem)
                obs_Y.append(obs_Y_elem)
                obs_elem = []
                obs_Y_elem = []
            if idx % 15 == 12:
                obs_elem = []
                obs_Y_elem = []

    return obs, obs_Y, obs_map
