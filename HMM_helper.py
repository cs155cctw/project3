########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 HMM helper
########################################

import re
import random
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import animation
from matplotlib.animation import FuncAnimation


####################
# WORDCLOUD FUNCTIONS
####################

def mask():
    # Parameters.
    r = 128
    d = 2 * r + 1

    # Get points in a circle.
    y, x = np.ogrid[-r:d-r, -r:d-r]
    circle = (x**2 + y**2 <= r**2)

    # Create mask.
    mask = 255 * np.ones((d, d), dtype=np.uint8)
    mask[circle] = 0

    return mask

def text_to_wordcloud(text, max_words=50, title='', show=True):
    plt.close('all')

    # Generate a wordcloud image.
    wordcloud = WordCloud(random_state=0,
                          max_words=max_words,
                          background_color='white',
                          mask=mask()).generate(text)

    # Show the image.
    if show:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=24)
        plt.show()

    return wordcloud

def states_to_wordclouds(hmm, obs_map, max_words=50, show=True):
    # Initialize.
    M = 100000
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = []

    # Generate a large emission.
    emission, states = hmm.generate_emission_nWords(M)

    # For each state, get a list of observations that have been emitted
    # from that state.
    obs_count = []
    for i in range(n_states):
        obs_lst = np.array(emission)[np.where(np.array(states) == i)[0]]
        obs_count.append(obs_lst)

    # For each state, convert it into a wordcloud.
    for i in range(n_states):
        obs_lst = obs_count[i]
        sentence = [obs_map_r[j] for j in obs_lst]
        sentence_str = ' '.join(sentence)

        wordclouds.append(text_to_wordcloud(sentence_str, max_words=max_words, title='State %d' % i, show=show))

    return wordclouds


####################
# HMM FUNCTIONS
####################

def parse_observations(text):
    # Convert text to dataset.
    lines = [line.split() for line in text.split('\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_map = {}

    for line in lines:
        obs_elem = []
        
        for word in line:
            word = re.sub(r'[^\w]', '', word).lower()
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
            
            # Add the encoded word.
            obs_elem.append(obs_map[word])
        
        # Add the encoded sequence.
        obs.append(obs_elem)

    return obs, obs_map

def obs_map_reverser(obs_map):
    obs_map_r = {}

    for key in obs_map:
        obs_map_r[obs_map[key]] = key

    return obs_map_r

def sample_sentence(hmm, obs_map, print_syllable = False, n_syllable = 10):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.generate_emission(obs_map_r, n_syllable)
    sentence = [obs_map_r[i] for i in emission]
    if print_syllable == False:
        sentence = []
        sentence = [obs_map_r[i].split('_')[0] for i in emission]
    return ' '.join(sentence).capitalize()

def sample_sentence_multipleModel(hmm_list, obs_map, print_syllable = False, n_syllable = 10):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    # emission, states = hmm.generate_emission(n_syllable)
    
    emissions = []
    syllable_num = 0.0
    
    probabilities = np.zeros(hmm_list[0].D)
    emission_index = range(hmm_list[0].D)
    
    states_previous = []
    multiply_factor = np.ones(hmm_list[0].D)
    for idx_hmm in range(len(hmm_list)):
        states_index = range(hmm_list[idx_hmm].L)
        state = random.choices(states_index, hmm_list[idx_hmm].A_start)[0]
        states_previous.append(state)
        
        for idx_obs in range(hmm_list[idx_hmm].D):
            probabilities[idx_obs] += hmm_list[idx_hmm].O[state][idx_obs]
            if (idx_hmm == 0 and hmm_list[0].O[state][idx_obs] == 0.0) or (obs_map_r[idx_obs].split('_')[1][0] == 'E'):
                multiply_factor[idx_obs] = 0.0
    
    sum_P = 0.0
    for idx_obs in range(hmm_list[idx_hmm].D):
        probabilities[idx_obs] = probabilities[idx_obs] * multiply_factor[idx_obs]
        sum_P += probabilities[idx_obs]
    if sum_P > 0:
        for idx_obs in range(hmm_list[idx_hmm].D):
            probabilities[idx_obs] = probabilities[idx_obs]/sum_P
            
    emission_rand_index = random.choices(emission_index, probabilities)[0]
    while obs_map_r[emission_rand_index].split('_')[1][0] == 'E':
        emission_rand_index = random.choices(emission_index, probabilities)[0]
    emissions.append(emission_rand_index)
    syllable_num += int(obs_map_r[emission_rand_index].split('_')[1][0])
    
    while syllable_num < n_syllable:
        probabilities = np.zeros(hmm_list[0].D)
        multiply_factor = np.ones(hmm_list[0].D)
        for idx_hmm in range(len(hmm_list)):
            states_index = range(hmm_list[idx_hmm].L)
            state = random.choices(states_index,hmm_list[idx_hmm].A[states_previous[idx_hmm]])[0]
            states_previous[idx_hmm] = state
            
            for idx_obs in range(hmm_list[idx_hmm].D):
                probabilities[idx_obs] += hmm_list[idx_hmm].O[state][idx_obs]
                if idx_hmm == 0 and hmm_list[0].O[state][idx_obs] == 0.0:
                    multiply_factor[idx_obs] = 0.0
                syllable_num_this = 0
                if obs_map_r[idx_obs].split('_')[1][0] == 'E':
                    syllable_num_this = int(obs_map_r[idx_obs].split('_')[1][1])
                    if syllable_num_this + syllable_num != 10:
                        multiply_factor[idx_obs] = 0.0
                else:
                    syllable_num_this = int(obs_map_r[idx_obs].split('_')[1][0])
                    if syllable_num_this + syllable_num > 10:
                        multiply_factor[idx_obs] = 0.0
        sum_P = 0.0
        for idx_obs in range(hmm_list[idx_hmm].D):
            probabilities[idx_obs] = probabilities[idx_obs] * multiply_factor[idx_obs]
            sum_P = sum_P + probabilities[idx_obs]
        if sum_P > 0:
            for idx_obs in range(hmm_list[idx_hmm].D):
                probabilities[idx_obs] = probabilities[idx_obs]/sum_P
        
        emission_rand_index = random.choices(emission_index, probabilities)[0]
        emissions.append(emission_rand_index)
        if obs_map_r[emission_rand_index].split('_')[1][0] == 'E':
            syllable_num = syllable_num + int(obs_map_r[emission_rand_index].split('_')[1][1])
        else:
            syllable_num = syllable_num + int(obs_map_r[emission_rand_index].split('_')[1][0])
        
    
    sentence = [obs_map_r[i] for i in emissions]
    if print_syllable == False:
        sentence = []
        sentence = [obs_map_r[i].split('_')[0] for i in emissions]

    return ' '.join(sentence).capitalize()

def sample_sentence_multipleModel_rythme(hmm_list, obs_map, rthyme_pair_lib, print_syllable = False, n_syllable = 10):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence_1.
    # emission, states = hmm.generate_emission(n_syllable)
    
    emissions_1 = []
    emissions_2 = []
    syllable_num_1 = 0.0
    syllable_num_2 = 0.0
    
    probabilities_1 = np.zeros(hmm_list[0].D)
    probabilities_2 = np.zeros(hmm_list[0].D)
    emission_index_1 = range(hmm_list[0].D)
    emission_index_2 = range(hmm_list[0].D)
    
    probabilities_1_start = []
    probabilities_2_start = []
    
    emission_index_1_start = []
    emission_index_2_start = []
    
    states_previous_1 = []
    states_previous_2 = []
    
    multiply_factor_1 = np.ones(hmm_list[0].D)
    multiply_factor_2 = np.ones(hmm_list[0].D)
    
    multiply_factor_1_start = []
    multiply_factor_2_start = []
    
    for idx_hmm in range(len(hmm_list)):
        states_index = range(hmm_list[idx_hmm].L)
        state = random.choices(states_index, hmm_list[idx_hmm].A_start)[0]
        states_previous_1.append(state)
        
        for idx_obs in range(hmm_list[idx_hmm].D):
            probabilities_1[idx_obs] += hmm_list[idx_hmm].O[state][idx_obs]
            if (idx_hmm == 0 and hmm_list[0].O[state][idx_obs] == 0.0) or (obs_map_r[idx_obs].split('_')[1][0] == 'E') or (not np.isin(idx_obs, rthyme_pair_lib)):
                multiply_factor_1[idx_obs] = 0.0
            else:
                emission_index_1_start.append(idx_obs)
                multiply_factor_1_start.append(1.0)
                probabilities_1_start.append(hmm_list[idx_hmm].O[state][idx_obs])
                
    sum_P_1 = 0.0
    for idx_obs in range(len(probabilities_1_start)):
        probabilities_1_start[idx_obs] = probabilities_1_start[idx_obs] * multiply_factor_1_start[idx_obs]
        sum_P_1 += probabilities_1_start[idx_obs]
    if sum_P_1 > 0:
        for idx_obs in range(len(probabilities_1_start)):
            probabilities_1_start[idx_obs] = probabilities_1_start[idx_obs]/sum_P_1
    emission_1_rand_index = random.choices(np.array(emission_index_1_start), np.array(probabilities_1_start))[0]
    emissions_1.append(emission_1_rand_index)
    if obs_map_r[emission_1_rand_index].split('_')[1][0] == 'E':
        syllable_num_1 += int(obs_map_r[emission_1_rand_index].split('_')[1][1])
    else:
        syllable_num_1 += int(obs_map_r[emission_1_rand_index].split('_')[1][0])
    
    #now determine the first word for the second line
    allPair = np.where(np.array(rthyme_pair_lib) == emission_1_rand_index)
    emission_2_candidates = []
    for idx in range(len(allPair[0])):
        emission_2_candidates.append(rthyme_pair_lib[allPair[0][idx]][1-allPair[1][idx]])
    emission_2_rand_index = random.choice(emission_2_candidates)
    emissions_2.append(emission_2_rand_index)
    if obs_map_r[emission_2_rand_index].split('_')[1][0] == 'E':
        syllable_num_2 += int(obs_map_r[emission_2_rand_index].split('_')[1][1])
    else:
        syllable_num_2 += int(obs_map_r[emission_2_rand_index].split('_')[1][0])
    #now determine the state for the first word
    for idx_hmm in range(len(hmm_list)):
        states_index = range(hmm_list[idx_hmm].L)
        probabilities_states = np.array(hmm_list[idx_hmm].O)[:,emission_2_rand_index]
        sum_P_states = np.sum(probabilities_states)
        probabilities_states = probabilities_states/sum_P_states
        state = random.choices(states_index, probabilities_states)[0]
        states_previous_2.append(state)
    
    while syllable_num_1 < n_syllable:
        probabilities_1 = np.zeros(hmm_list[0].D)
        multiply_factor_1 = np.ones(hmm_list[0].D)
        for idx_hmm in range(len(hmm_list)):
            states_index = range(hmm_list[idx_hmm].L)
            state = random.choices(states_index,hmm_list[idx_hmm].A[states_previous_1[idx_hmm]])[0]
            states_previous_1[idx_hmm] = state
            
            for idx_obs in range(hmm_list[idx_hmm].D):
                probabilities_1[idx_obs] += hmm_list[idx_hmm].O[state][idx_obs]
                if idx_hmm == 0 and hmm_list[0].O[state][idx_obs] == 0.0:
                    multiply_factor_1[idx_obs] = 0.0
                syllable_num_1_this = 0
                if obs_map_r[idx_obs].split('_')[1][0] == 'E':
                    multiply_factor_1[idx_obs] = 0.0
                else:
                    syllable_num_1_this = int(obs_map_r[idx_obs].split('_')[1][0])
                    if syllable_num_1_this + syllable_num_1 > 10:
                        multiply_factor_1[idx_obs] = 0.0
        sum_P_1 = 0.0
        for idx_obs in range(hmm_list[idx_hmm].D):
            probabilities_1[idx_obs] = probabilities_1[idx_obs] * multiply_factor_1[idx_obs]
            sum_P_1 = sum_P_1 + probabilities_1[idx_obs]
        if sum_P_1 > 0:
            for idx_obs in range(hmm_list[idx_hmm].D):
                probabilities_1[idx_obs] = probabilities_1[idx_obs]/sum_P_1
        
        emission_1_rand_index = random.choices(emission_index_1, probabilities_1)[0]
        emissions_1.append(emission_1_rand_index)
        if obs_map_r[emission_1_rand_index].split('_')[1][0] == 'E':
            syllable_num_1 = syllable_num_1 + int(obs_map_r[emission_1_rand_index].split('_')[1][1])
        else:
            syllable_num_1 = syllable_num_1 + int(obs_map_r[emission_1_rand_index].split('_')[1][0])
    while syllable_num_2 < n_syllable:
        probabilities_2 = np.zeros(hmm_list[0].D)
        multiply_factor_2 = np.ones(hmm_list[0].D)
        for idx_hmm in range(len(hmm_list)):
            states_index = range(hmm_list[idx_hmm].L)
            state = random.choices(states_index,hmm_list[idx_hmm].A[states_previous_2[idx_hmm]])[0]
            states_previous_2[idx_hmm] = state
            
            for idx_obs in range(hmm_list[idx_hmm].D):
                probabilities_2[idx_obs] += hmm_list[idx_hmm].O[state][idx_obs]
                if idx_hmm == 0 and hmm_list[0].O[state][idx_obs] == 0.0:
                    multiply_factor_2[idx_obs] = 0.0
                syllable_num_2_this = 0
                if obs_map_r[idx_obs].split('_')[1][0] == 'E':
                    multiply_factor_2[idx_obs] = 0.0
                else:
                    syllable_num_2_this = int(obs_map_r[idx_obs].split('_')[1][0])
                    if syllable_num_2_this + syllable_num_2 > 10:
                        multiply_factor_2[idx_obs] = 0.0
        sum_P_2 = 0.0
        for idx_obs in range(hmm_list[idx_hmm].D):
            probabilities_2[idx_obs] = probabilities_2[idx_obs] * multiply_factor_2[idx_obs]
            sum_P_2 = sum_P_2 + probabilities_2[idx_obs]
        if sum_P_2 > 0:
            for idx_obs in range(hmm_list[idx_hmm].D):
                probabilities_2[idx_obs] = probabilities_2[idx_obs]/sum_P_2
        
        emission_2_rand_index = random.choices(emission_index_2, probabilities_2)[0]
        emissions_2.append(emission_2_rand_index)
        if obs_map_r[emission_2_rand_index].split('_')[1][0] == 'E':
            syllable_num_2 = syllable_num_2 + int(obs_map_r[emission_2_rand_index].split('_')[1][1])
        else:
            syllable_num_2 = syllable_num_2 + int(obs_map_r[emission_2_rand_index].split('_')[1][0])
    
    emissions_1.reverse()
    emissions_2.reverse()
    sentence_1 = [obs_map_r[i] for i in emissions_1]
    sentence_2 = [obs_map_r[i] for i in emissions_2]
    if print_syllable == False:
        sentence_1 = []
        sentence_2 = []
        sentence_1 = [obs_map_r[i].split('_')[0] for i in emissions_1]
        sentence_2 = [obs_map_r[i].split('_')[0] for i in emissions_2]
    
    return ' '.join(sentence_1).capitalize(), ' '.join(sentence_2).capitalize()


def sample_sentence_rythme(hmm, obs_map, rthyme_pair_lib, print_syllable = False, n_syllable = 10):

    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission1,states1,emission2,states2 = hmm.generate_emission_reverse(obs_map_r, rthyme_pair_lib, n_syllable)
    sentence1 = [obs_map_r[i] for i in emission1]
    sentence2 = [obs_map_r[i] for i in emission2]
    if print_syllable == False:
        sentence1 = []
        sentence1 = [obs_map_r[i].split('_')[0] for i in emission1]
        sentence2 = []
        sentence2 = [obs_map_r[i].split('_')[0] for i in emission2]
        
    return ' '.join(sentence1).capitalize(),' '.join(sentence2).capitalize() 



def sample_poem(hmm, obs_map):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.generate_emission(n_words)
    sentence = [obs_map_r[i] for i in emission]

    return ' '.join(sentence).capitalize() + '...'




####################
# HMM VISUALIZATION FUNCTIONS
####################

def visualize_sparsities(hmm, O_max_cols=50, O_vmax=0.1):
    plt.close('all')
    plt.set_cmap('viridis')

    # Visualize sparsity of A.
    plt.imshow(hmm.A, vmax=1.0)
    plt.colorbar()
    plt.title('Sparsity of A matrix')
    plt.show()

    # Visualize parsity of O.
    plt.imshow(np.array(hmm.O)[:, :O_max_cols], vmax=O_vmax, aspect='auto')
    plt.colorbar()
    plt.title('Sparsity of O matrix')
    plt.show()


####################
# HMM ANIMATION FUNCTIONS
####################

def animate_emission(hmm, obs_map, M=8, height=12, width=12, delay=1):
    # Parameters.
    lim = 1200
    text_x_offset = 40
    text_y_offset = 80
    x_offset = 580
    y_offset = 520
    R = 420
    r = 100
    arrow_size = 20
    arrow_p1 = 0.03
    arrow_p2 = 0.02
    arrow_p3 = 0.06
    
    # Initialize.
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = states_to_wordclouds(hmm, obs_map, max_words=20, show=False)

    # Initialize plot.    
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.grid('off')
    plt.axis('off')
    ax.set_xlim([0, lim])
    ax.set_ylim([0, lim])

    # Plot each wordcloud.
    for i, wordcloud in enumerate(wordclouds):
        x = x_offset + int(R * np.cos(np.pi * 2 * i / n_states))
        y = y_offset + int(R * np.sin(np.pi * 2 * i / n_states))
        ax.imshow(wordcloud.to_array(), extent=(x - r, x + r, y - r, y + r), aspect='auto', zorder=-1)

    # Initialize text.
    text = ax.text(text_x_offset, lim - text_y_offset, '', fontsize=24)
        
    # Make the arrows.
    zorder_mult = n_states ** 2 * 100
    arrows = []
    for i in range(n_states):
        row = []
        for j in range(n_states):
            # Arrow coordinates.
            x_i = x_offset + R * np.cos(np.pi * 2 * i / n_states)
            y_i = y_offset + R * np.sin(np.pi * 2 * i / n_states)
            x_j = x_offset + R * np.cos(np.pi * 2 * j / n_states)
            y_j = y_offset + R * np.sin(np.pi * 2 * j / n_states)
            
            dx = x_j - x_i
            dy = y_j - y_i
            d = np.sqrt(dx**2 + dy**2)

            if i != j:
                arrow = ax.arrow(x_i + (r/d + arrow_p1) * dx + arrow_p2 * dy,
                                 y_i + (r/d + arrow_p1) * dy + arrow_p2 * dx,
                                 (1 - 2 * r/d - arrow_p3) * dx,
                                 (1 - 2 * r/d - arrow_p3) * dy,
                                 color=(1 - hmm.A[i][j], ) * 3,
                                 head_width=arrow_size, head_length=arrow_size,
                                 zorder=int(hmm.A[i][j] * zorder_mult))
            else:
                arrow = ax.arrow(x_i, y_i, 0, 0,
                                 color=(1 - hmm.A[i][j], ) * 3,
                                 head_width=arrow_size, head_length=arrow_size,
                                 zorder=int(hmm.A[i][j] * zorder_mult))

            row.append(arrow)
        arrows.append(row)

    emission, states = hmm.generate_emission(M)

    def animate(i):
        if i >= delay:
            i -= delay

            if i == 0:
                arrows[states[0]][states[0]].set_color('red')
            elif i == 1:
                arrows[states[0]][states[0]].set_color((1 - hmm.A[states[0]][states[0]], ) * 3)
                arrows[states[i - 1]][states[i]].set_color('red')
            else:
                arrows[states[i - 2]][states[i - 1]].set_color((1 - hmm.A[states[i - 2]][states[i - 1]], ) * 3)
                arrows[states[i - 1]][states[i]].set_color('red')

            # Set text.
            text.set_text(' '.join([obs_map_r[e] for e in emission][:i+1]).capitalize())

            return arrows + [text]

    # Animate!
    print('\nAnimating...')
    anim = FuncAnimation(fig, animate, frames=M+delay, interval=1000)

    return anim

    # honestly this function is so jank but who even fuckin cares
    # i don't even remember how or why i wrote this mess
    # no one's gonna read this
    # hey if you see this tho hmu on fb let's be friends
