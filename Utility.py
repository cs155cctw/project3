
import os
import re
import numpy as np


def get_poem_sequence(filename, option):
    '''
    option: poem, stanza, line
    output: an array of sequence, each sequence is an array of encoded word for a poem, a stanza, or a line
    '''
    obs = []
    obs_map = []
    # Convert text to dataset.
    text = open(os.path.join(os.getcwd(), filename)).read()

    lines = [line.split() for line in text.split('\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_map = {}
    
    #print(len(lines))
    #print(lines[1])
    if option == 'line':
        for line in lines:
            obs_elem = []
            if len(line) < 2:
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

            # Add the encoded sequence.
            obs.append(obs_elem)

    return obs, obs_map
