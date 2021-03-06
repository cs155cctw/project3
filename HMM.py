########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    Output sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        # Calculate initial prefixes and probabilities.
        for curr in range(self.L):
            probs[1][curr] = self.A_start[curr] * self.O[curr][x[0]]
            seqs[1][curr] = str(curr)

        # Calculate best prefixes and probabilities throughout sequence.
        for t in range(2, M + 1):
            # Iterate over all possible current states.
            for curr in range(self.L):
                max_prob = float("-inf")
                max_prefix = ''

                # Iterate over all possible previous states to find one
                # that would maximize the probability of the current state.
                for prev in range(self.L):
                    curr_prob = probs[t - 1][prev] \
                                * self.A[prev][curr] \
                                * self.O[curr][x[t - 1]]

                    # Continually update max probability and prefix.
                    if curr_prob >= max_prob:
                        max_prob = curr_prob
                        max_prefix = seqs[t - 1][prev]

                # Store the max probability and prefix.
                probs[t][curr] = max_prob
                seqs[t][curr] = max_prefix + str(curr)

        # Find the index of the max probability of a sequence ending in x^M
        # and the corresponding output sequence.
        max_i = max(enumerate(probs[-1]), key=lambda x: x[1])[0]
        max_seq = seqs[-1][max_i]

        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Note that alpha_j(0) is already correct for all j's.
        # Calculate alpha_j(1) for all j's.
        for curr in range(self.L):
            alphas[1][curr] = self.A_start[curr] * self.O[curr][x[0]]

        # Calculate alphas throughout sequence.
        for t in range(1, M):
            # Iterate over all possible current states.
            for curr in range(self.L):
                prob = 0

                # Iterate over all possible previous states to accumulate
                # the probabilities of all paths from the start state to
                # the current state.
                for prev in range(self.L):
                    prob += alphas[t][prev] \
                            * self.A[prev][curr] \
                            * self.O[curr][x[t]]

                # Store the accumulated probability.
                alphas[t + 1][curr] = prob

            if normalize:
                norm = sum(alphas[t + 1])
                for curr in range(self.L):
                    alphas[t + 1][curr] /= norm

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize initial betas.
        for curr in range(self.L):
            betas[-1][curr] = 1

        # Calculate betas throughout sequence.
        for t in range(-1, -M - 1, -1):
            # Iterate over all possible current states.
            for curr in range(self.L):
                prob = 0

                # Iterate over all possible next states to accumulate
                # the probabilities of all paths from the end state to
                # the current state.
                for nxt in range(self.L):
                    if t == -M:
                        prob += betas[t][nxt] \
                                * self.A_start[nxt] \
                                * self.O[nxt][x[t]]

                    else:
                        prob += betas[t][nxt] \
                                * self.A[curr][nxt] \
                                * self.O[nxt][x[t]]

                # Store the accumulated probability.
                betas[t - 1][curr] = prob

            if normalize:
                norm = sum(betas[t - 1])
                for curr in range(self.L):
                    betas[t - 1][curr] /= norm

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        for curr in range(self.L):
            for nxt in range(self.L):
                num = 0.
                den = 0.

                for i in range(len(X)):
                    x = X[i]
                    y = Y[i]
                    M = len(x)
        
                    num += len([1 for i in range(M - 1) \
                                if y[i] == curr and y[i + 1] == nxt])
                    den += len([1 for i in range(M - 1) if y[i] == curr])

                self.A[curr][nxt] = num / den

        # Calculate each element of O using the M-step formulas.
        for curr in range(self.L):
            for xt in range(self.D):
                num = 0.
                den = 0.

                for i in range(len(X)):
                    x = X[i]
                    y = Y[i]
                    M = len(x)
        
                    num += len([1 for i in range(M) \
                                if y[i] == curr and x[i] == xt])
                    den += len([1 for i in range(M) if y[i] == curr])

                self.O[curr][xt] = num / den
                
  
    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        # Note that a comment starting with 'E' refers to the fact that
        # the code under the comment is part of the E-step.

        # Similarly, a comment starting with 'M' refers to the fact that
        # the code under the comment is part of the M-step.

        for iteration in range(1, N_iters + 1):
            if iteration % 10 == 0:
                print("Iteration: " + str(iteration))

            # Numerator and denominator for the update terms of A and O.
            A_num = [[0. for i in range(self.L)] for j in range(self.L)]
            O_num = [[0. for i in range(self.D)] for j in range(self.L)]
            A_den = [0. for i in range(self.L)]
            O_den = [0. for i in range(self.L)]

            # For each input sequence:
            for x in X:
                M = len(x)
                # Compute the alpha and beta probability vectors.
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                # E: Update the expected observation probabilities for a
                # given (x, y).
                # The i^th index is P(y^t = i, x).
                for t in range(1, M + 1):
                    P_curr = [0. for _ in range(self.L)]
                    
                    for curr in range(self.L):
                        P_curr[curr] = alphas[t][curr] * betas[t][curr]

                    # Normalize the probabilities.
                    norm = sum(P_curr)
                    for curr in range(len(P_curr)):
                        P_curr[curr] /= norm

                    for curr in range(self.L):
                        if t != M:
                            A_den[curr] += P_curr[curr]
                        O_den[curr] += P_curr[curr]
                        O_num[curr][x[t - 1]] += P_curr[curr]

                # E: Update the expectedP(y^j = a, y^j+1 = b, x) for given (x, y)
                for t in range(1, M):
                    P_curr_nxt = [[0. for _ in range(self.L)] for _ in range(self.L)]

                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] = alphas[t][curr] \
                                                    * self.A[curr][nxt] \
                                                    * self.O[nxt][x[t]] \
                                                    * betas[t + 1][nxt]

                    # Normalize:
                    norm = 0
                    for lst in P_curr_nxt:
                        norm += sum(lst)
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] /= norm

                    # Update A_num
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_num[curr][nxt] += P_curr_nxt[curr][nxt]

            for curr in range(self.L):
                for nxt in range(self.L):
                    self.A[curr][nxt] = A_num[curr][nxt] / A_den[curr]

            for curr in range(self.L):
                for xt in range(self.D):
                    self.O[curr][xt] = O_num[curr][xt] / O_den[curr]


    def generate_emission(self,obs_map_r, M=10):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        syllable_num = 0.0
        

        states_index = range(self.L)
        emission_full_index = range(self.D)

        #emission_full_index [emission_full_index == 6] = ''
        emission_index = []
        
        for i in range(self.D):
         
            if obs_map_r[i].split('_')[1][0] != 'E':
                emission_index.append(i)
            
            

       
        states.append(random.choices(states_index, self.A_start)[0])
        O_mat = [self.O[states[0]][i] for i in emission_index]
        emission_rand_index = random.choices(emission_index, O_mat )[0]
        emission.append(emission_rand_index)
        #print(obs_map_r[emission_rand_index].split('_')[1][0])
        syllable_num += int(obs_map_r[emission_rand_index].split('_')[1][0])
       
        wordnum = 1
        
        
                
        while syllable_num < M:
            
            #print(emission)
            #print(states)
            states.append(random.choices(states_index,self.A[states[wordnum-1]])[0])
            emission_rand_index= random.choices(emission_full_index, self.O[states[wordnum]])[0]

            if obs_map_r[emission_rand_index].split('_')[1][0] == 'E'and int(obs_map_r[emission_rand_index].split('_')[1][1]) + syllable_num == 10:                
                emission.append(emission_rand_index)
                syllable_num += int(obs_map_r[emission_rand_index].split('_')[1][0])
                wordnum += 1
                break 
            elif obs_map_r[emission_rand_index].split('_')[1][0] != 'E' and int(obs_map_r[emission_rand_index].split('_')[1][0])+ syllable_num == 10:                 
                emission.append(emission_rand_index)
                syllable_num += int(obs_map_r[emission_rand_index].split('_')[1][0])
                wordnum += 1
                break 
            else:
                emission_index = []
                for i in range(self.D):
                    
                    if obs_map_r[i].split('_')[1][0] != 'E' and int(obs_map_r[i].split('_')[1][0]) + syllable_num <= 10:
                        emission_index.append(i)
                
                O_mat = [self.O[states[wordnum]][i] for i in emission_index]
                emission_rand_index = random.choices(emission_index, O_mat)[0]
                emission.append(emission_rand_index)
                syllable_num += int(obs_map_r[emission_rand_index].split('_')[1][0])
                wordnum += 1
                   
                
                       
                #if syllable_num == 10:
                    #break
                
                #elif syllable_num < 10:
                    #continue 
       
 
            

        #print(syllable_num)
        return emission, states
    
    def generate_emission_nWords(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        states_index = range(self.L)
        emission_index = range(self.D)

        states.append(random.choices(states_index, self.A_start)[0])
        emission.append(random.choices(emission_index, self.O[states[0]])[0])
        #states.append(np.random.choice(self.L, 1, p = self.A_start)[0])
        #emission.append(np.random.choice(self.D, 1, p = self.O[states[0]])[0])
        '''
        max_index = 0
        prob = 0
        for emission_index in range(self.D):
            if prob < self.O[states[0]][emission_index]:
               prob = self.O[states[0]][emission_index]
               max_index = emission_index
        emission.append(max_index)
        '''

        for sequence_index in range(M):
            if sequence_index > 0:
                '''
                max_index = 0
                prob = 0
                for state_index in range(self.L):
                    if prob < self.A[states[sequence_index-1]][state_index]:
                       prob = self.A[states[sequence_index-1]][state_index]
                       max_index = state_index
                states.append(max_index)
                prob = 0
                max_index = 0
                for emission_index in range(self.D):
                    if prob < self.O[states[sequence_index]][emission_index]:
                       prob = self.O[states[sequence_index]][emission_index]
                       max_index = emission_index
                emission.append(max_index)
                '''
                states.append(random.choices(states_index,self.A[states[sequence_index-1]])[0])
                emission.append(random.choices(emission_index, self.O[states[sequence_index]])[0])
                #emission.append(3)
                #states.append(np.random.choice(self.L, 1, p = self.A[states[sequence_index-1]])[0])
                #emission.append(np.random.choice(self.D, 1, p = self.O[states[sequence_index]])[0])



        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2F)
        ###
        ###
        ###

        return emission, states

    def generate_emission_reverse(self,obs_map_r,rthyme_pair_lib, M=10):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission_1 = []
        states_1 = []
        syllable_num_1 = 0.0
        
        emission_2 = []
        states_2 = []
        syllable_num_2 = 0.0

        states_1_index = range(self.L)
        states_2_index = range(self.L)
        
        emission_1_index = [] 
        emission_2_index = []        
        
        
        for i in range(self.D):
            if obs_map_r[i].split('_')[1][0] != 'E':
                emission_1_index.append(i)
                emission_2_index.append(i)
        
    
        rthyme_pair_loc_1 = random.randint(0, 1)
        rthyme_pair = random.choice(rthyme_pair_lib)
        if rthyme_pair_loc_1 ==0:
            rthyme_pair_loc_2 = 1
        else:
            rthyme_pair_loc_2 = 0
            
            
        states_1.append(random.choices(states_1_index, self.A_start)[0])
        states_2.append(random.choices(states_2_index, self.A_start)[0])
        
        emission_1_rand_index = int(rthyme_pair[rthyme_pair_loc_1])
        emission_2_rand_index = int(rthyme_pair[rthyme_pair_loc_2])

        emission_1.append(emission_1_rand_index)
        emission_2.append(emission_2_rand_index)
        
        syllable_num_1 += int(obs_map_r[emission_1_rand_index].split('_')[1][-1])
        syllable_num_2 += int(obs_map_r[emission_2_rand_index].split('_')[1][-1])
        
        wordnum_1 = 1
        wordnum_2 = 1
        
        
               
        while syllable_num_1 < M:
        
            states_1.append(random.choices(states_1_index,self.A[states_1[wordnum_1-1]])[0])
            
            emission_1_index = []
            for i in range(self.D):
                if obs_map_r[i].split('_')[1][0] != 'E' and int(obs_map_r[i].split('_')[1][0]) + syllable_num_2 <= 10:
                    emission_1_index.append(i)
          
            O_mat_1 = [self.O[states_1[wordnum_1]][i] for i in emission_1_index]
           
            
            emission_1_rand_index = random.choices(emission_1_index, O_mat_1)[0]
            emission_1.append(emission_1_rand_index)
            syllable_num_1 += int(obs_map_r[emission_1_rand_index].split('_')[1][0])
            wordnum_1 += 1
                   
               
                
        while syllable_num_2 < M:
            

            states_2.append(random.choices(states_2_index,self.A[states_2[wordnum_2-1]])[0])
           
            emission_2_index = []
            for i in range(self.D):
                if obs_map_r[i].split('_')[1][0] != 'E' and int(obs_map_r[i].split('_')[1][0]) + syllable_num_2 <= 10:
                    emission_2_index.append(i)
                
            O_mat_2 = [self.O[states_2[wordnum_2]][i] for i in emission_2_index]
            emission_2_rand_index = random.choices(emission_2_index, O_mat_2)[0]
            emission_2.append(emission_2_rand_index)
            syllable_num_2 += int(obs_map_r[emission_2_rand_index].split('_')[1][0])
            wordnum_2 += 1

        emission_1.reverse()
        states_1.reverse()
        emission_2.reverse()
        states_2.reverse()
        
        
        return emission_1, states_1, emission_2, states_2


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the output sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(0) gives the probability of the output sequence. Summing
        # this over all states and then normalizing gives the total
        # probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum([betas[1][k] * self.A_start[k] * self.O[k][x[0]] \
            for k in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters, seedn):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    random.seed(seedn)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM

def merge_HMM (hmm_list):
    AA = [[0 for i in range(hmm_list[0].L)] for j in range(hmm_list[0].L)];
    OO = O = [[0 for i in range(hmm_list[0].D)] for j in range(hmm_list[0].L)];
    
    for index in range(len(hmm_list)):
        for state_index in range(hmm_list[0].L):
            for emission_index in range(hmm_list[0].D):
                OO[state_index][emission_index] += hmm_list[index].O[state_index][emission_index]/len(hmm_list)
            for state_index_next in range(hmm_list[0].L):
                AA[state_index][state_index_next] += hmm_list[index].A[state_index][state_index_next]/len(hmm_list)
    HMM = HiddenMarkovModel(AA, OO)
    return HMM
    