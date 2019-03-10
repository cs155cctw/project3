import os
import re
import numpy as np


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
    obs_map = {}
    obs_elem = []


    for idx in range(len(lines)):
        line = lines[idx]
        # ignore poem number
        if idx % 15 < 1:
            continue
        for word in line:
            word = re.sub(r'[^\w]', '', word).lower()
            if word == '':
                continue
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
            # Add the encoded word.
            obs_elem.append(obs_map[word])

        if option == 'line':
            obs.append(obs_elem)
            obs_elem = []

        if option == 'poem':
            if idx % 15 == 14:
                # Add the encoded sequence.
                obs.append(obs_elem)
                # initialzie the sequence
                obs_elem = []

        if option == 'stanza1':
            if idx % 15 == 4:
                obs.append(obs_elem)
                obs_elem = []
            if idx % 15 == 14:
                obs_elem = []

        if option == 'stanza2':
            if idx % 15 == 8:
                obs.append(obs_elem)
                obs_elem = []
            if idx % 15 == 4:
                obs_elem = []

        if option == 'stanza3':
            if idx % 15 == 12:
                obs.append(obs_elem)
                obs_elem = []
            if idx % 15 == 8:
                obs_elem = []

        if option == 'stanza4':
            if idx % 15 == 14:
                obs.append(obs_elem)
                obs_elem = []
            if idx % 15 == 12:
                obs_elem = []

    return obs, obs_map
