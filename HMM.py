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
            max_seq:    State sequence corresponding to x with the highest
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

        # prob[0] is the start state, let the entried remain zero
        # Initialize probs[1] and seqs[1] first
        for index in range(self.L):
            probs[1][index] = self.A_start[index]*self.O[index][x[0]]
            seqs[1][index] = str(index)

        for sequence_index in range(M):
            if sequence_index > 0:
               for state_index in range(self.L):
                   max_prob_temp = 0
                   max_index = 0
                   for last_sequence_state_index in range(self.L):
                       prob_temp = probs[sequence_index][last_sequence_state_index]*self.A[last_sequence_state_index][state_index]*self.O[state_index][x[sequence_index]]
                       if max_prob_temp < prob_temp:
                          max_prob_temp = prob_temp
                          max_index = last_sequence_state_index
                   probs[sequence_index+1][state_index] = max_prob_temp
                   seqs[sequence_index+1][state_index] = seqs[sequence_index][max_index]+str(state_index)
        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2A)
        ###
        ###
        ###
        prob_temp = 0
        for index in range(self.L):
            if prob_temp < probs[M][index]:
               prob_temp = probs[M][index]
               max_seq = seqs[M][index]
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
        # Intialize alpha[1] since its dependence on alpha[0] is special
        start_sum = 0
        for state_index in range(self.L):
            alphas[1][state_index] = self.O[state_index][x[0]]*1*self.A_start[state_index]
            start_sum += self.O[state_index][x[0]]*1*self.A_start[state_index]
        if normalize:
           for index in range(self.L):
               alphas[1][index] /= start_sum

        # compute from sequence with 2 words to those of M words
        for sequence_index in range(M):
            if sequence_index > 0:
               sequence_sum = 0 # in case of normalization
               for state_index in range(self.L): # loop through each possible state
                   alpha_last_sum = 0 # initialize the alpha_sum from previous sequence
                   for last_sequence_state_index in range(self.L):
                       alpha_last_sum += alphas[sequence_index][last_sequence_state_index]*self.A[last_sequence_state_index][state_index]
                   alphas[sequence_index+1][state_index] = self.O[state_index][x[sequence_index]]*alpha_last_sum
                   sequence_sum += self.O[state_index][x[sequence_index]]*alpha_last_sum
                # apply normalization if normalize == true after computation ends for each index
               if normalize:
                  for index in range(self.L):
                      alphas[sequence_index+1][index] /= sequence_sum        
        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bi)
        ###
        ###
        ###

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
        # Initialize betas[M]
        for index in range(self.L):
            betas[M][index] = 1
        if normalize:
           for index in range(self.L):
               betas[M][index] = 1./self.L
        # loop starts from M-1
        for index in range(M):
            if index > 0:
               sequence_index = M-index
               sequence_sum = 0
               for state_index in range(self.L):
                   for next_state_index in range(self.L):
                       betas[sequence_index][state_index] += betas[sequence_index+1][next_state_index]*self.A[state_index][next_state_index]*self.O[next_state_index][x[sequence_index]]
                   sequence_sum += betas[sequence_index][state_index]
               if normalize:
                  for index in range(self.L):
                      betas[sequence_index][index] /= sequence_sum




        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bii)
        ###
        ###
        ###

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
        for current_state in range(self.L):
            for next_state in range(self.L):
                numerator = 0
                denominator = 0
                for index in range(len(Y)):
                    for list_index in range(len(Y[index])-1):
                        if (Y[index][list_index] == current_state):
                            denominator += 1
                            if (Y[index][list_index+1] == next_state):
                               numerator += 1

                self.A[current_state][next_state] = numerator/denominator

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2C)
        ###
        ###
        ###

        # Calculate each element of O using the M-step formulas.
        for state_index in range(self.L):
            for emission_state in range(self.D):
                numerator = 0
                denominator = 0
                for index in range(len(Y)):
                    for list_index in range(len(Y[index])):
                        if (Y[index][list_index] == state_index):
                            denominator += 1
                            if (X[index][list_index] == emission_state):
                                numerator += 1
                self.O[state_index][emission_state] = numerator/denominator
        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2C)
        ###
        ###
        ###

        pass


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
        normalize = True
        for iteration in range(N_iters):
            if iteration%10 == 0:
                print('iteration '+str(iteration))
            P_y_xall = []
            P_ylast_ycurrent_xall = []

            for input_index in range(len(X)):
                xi = X[input_index]
                Mi = len(xi)
                alphas = self.forward(xi,normalize)
                betas = self.backward(xi,normalize)
                P_currentstate_xi = [[0. for _ in range(self.L)] for _ in range(Mi)]
                P_laststate_currentstate_xi = []
                denominator1 = [0. for _ in range(Mi)]
                denominator2 = [0. for _ in range(Mi-1)]

                # compute P(y^j=a,xi) and P(y^j=a,y^j+1=b,xi)
                for sequence_index in range(Mi):
                    for state_index in range(self.L):
                        denominator1[sequence_index] += alphas[sequence_index+1][state_index] * betas[sequence_index+1][state_index]
                
                for sequence_index in range(Mi):
                    for state_index in range(self.L):
                        P_currentstate_xi[sequence_index][state_index] = alphas[sequence_index+1][state_index]*betas[sequence_index+1][state_index]/denominator1[sequence_index]


                for sequence_index in range(Mi-1):
                    for state_index_a in range(self.L):
                        for state_index_b in range(self.L):
                            denominator2[sequence_index] += alphas[sequence_index+1][state_index_a]*betas[sequence_index+2][state_index_b]*\
                            self.O[state_index_b][xi[sequence_index+1]]*self.A[state_index_a][state_index_b]

                for sequence_index in range(Mi-1):
                    P_laststate_currentstate_xi.append([[0. for _ in range(self.L)] for _ in range(self.L)])
                    for last_state in range(self.L):
                        for current_state in range(self.L):
                            P_laststate_currentstate_xi[sequence_index][last_state][current_state] = alphas[sequence_index+1][last_state]*\
                            betas[sequence_index+2][current_state]*self.A[last_state][current_state]*self.O[current_state][xi[sequence_index+1]]/denominator2[sequence_index]

                P_y_xall.append(P_currentstate_xi)
                P_ylast_ycurrent_xall.append(P_laststate_currentstate_xi)

            # now start to update A and O
            for last_state_index in range(self.L):
                for current_state_index in range(self.L):

                    numerator = 0
                    denominator = 0
                    for index in range(len(X)):
                        Mi = len(X[index])
                        for sequence_index in range(Mi-1):
                            numerator +=  P_ylast_ycurrent_xall[index][sequence_index][last_state_index][current_state_index]
                            denominator += P_y_xall[index][sequence_index][last_state_index]
                    self.A[last_state_index][current_state_index] = numerator/denominator

            for state_index in range(self.L):
                for emission_index in range(self.D):

                    numerator = 0
                    denominator = 0
                    for index in range(len(X)):
                        Mi = len(X[index])
                        for sequence_index in range(Mi):
                            if (X[index][sequence_index] == emission_index):
                               numerator += P_y_xall[index][sequence_index][state_index]
                            denominator += P_y_xall[index][sequence_index][state_index]
                    self.O[state_index][emission_index] = numerator/denominator

        # checked for indexing: states: happy-0, mellow-1, sad-2, angry-3
        # genres: rock-0, pop-1, house-2, metal-3, folk-4, blues-5, dubstep-6, jazz-7, rap-8, classical-9
        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2D)
        ###
        ###
        ###

        pass


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
                   
                
                       
                if syllable_num + int(obs_map_r[emission_rand_index].split('_')[1][0]) == 10:
                    emission.append(emission_rand_index)
                    syllable_num += int(obs_map_r[emission_rand_index].split('_')[1][0])
                    wordnum += 1
                    break
                
                elif syllable_num + int(obs_map_r[emission_rand_index].split('_')[1][0]) < 10:
                    emission.append(emission_rand_index)
                    syllable_num += int(obs_map_r[emission_rand_index].split('_')[1][0])
                    wordnum += 1
                    continue 
       
 
            

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
        emission_1_full_index = range(self.D)
        emission_1_index = []       
                
        states_2_index = range(self.L)
        emission_2_full_index = range(self.D)
        emission_2_index = []


        #emission_full_index [emission_full_index == 6] = ''
        # first randomly select
        
    
        rthyme_pair_loc_1 = random.randint(0, 1)
        rthyme_pair = random.choice(rthyme_pair_lib)
        if rthyme_pair_loc_1 ==0:
            rthyme_pair_loc_2 = 1
        else:
            rthyme_pair_loc_2 = 0
            
            
        states_1.append(random.choices(states_1_index, self.A_start)[0])
        states_2.append(random.choices(states_2_index, self.A_start)[0])
        
        emission_rthyme_index_1 = int(rthyme_pair[rthyme_pair_loc_1])
        emission_rthyme_index_2 = int(rthyme_pair[rthyme_pair_loc_2])
        
      
        
        #print(self.O[states_1])
       
        #print(self.O[states_1])
        
        #print(len(self.O))
        #print(len(self.O[0]))
        #print(states_1_index)
        #print(emission_rthyme_index_1)
        #print(self.O[0][emission_rthyme_index_1])
        #print(self.O[1][emission_rthyme_index_1])
        #print(self.O[2][emission_rthyme_index_1])
        #print(self.O[3][emission_rthyme_index_1])
        #print(self.O[4][emission_rthyme_index_1])
        #print(self.O[5][emission_rthyme_index_1])
        #print(self.O[6][emission_rthyme_index_1])
        #print(self.O[7][emission_rthyme_index_1])
        #print(self.O[8][emission_rthyme_index_1])
        #print(self.O[9][emission_rthyme_index_1])
        
        
        O_mat_1 = [self.O[j][emission_rthyme_index_1]/sum(np.array(self.O)[:,emission_rthyme_index_1]) for j in states_1_index]
        O_mat_2 = [self.O[j][emission_rthyme_index_2]/sum(np.array(self.O)[:,emission_rthyme_index_2]) for j in states_2_index]
        


        print(len(emission_1_full_index))
        print(emission_rthyme_index_1)
        print(emission_rthyme_index_2)
        print(len(O_mat_1))
        emission_1_rand_index = random.choices(states_1_index, O_mat_1)[0]
        emission_2_rand_index = random.choices(states_2_index, O_mat_2)[0]
        
        emission_1.append(emission_1_rand_index)
        emission_2.append(emission_2_rand_index)
        
        syllable_num_1 += int(obs_map_r[emission_1_rand_index].split('_')[1][-1])
        syllable_num_2 += int(obs_map_r[emission_2_rand_index].split('_')[1][-1])
        
        wordnum_1 = 1
        wordnum_2 = 1
        
        
               
        while syllable_num_1 < M:
            
            #print(emission)
            #print(states)
            states_1.append(random.choices(states_1_index,self.A[states_1[wordnum_1-1]])[0])
            emission_1_rand_index= random.choices(emission_1_full_index, self.O[states_1[wordnum_1]])[0]

            if obs_map_r[emission_1_rand_index].split('_')[1][0] == 'E'and int(obs_map_r[emission_1_rand_index].split('_')[1][1]) + syllable_num_1 == 10:                
                emission_1.append(emission_1_rand_index)
                syllable_num_1 += int(obs_map_r[emission_1_rand_index].split('_')[1][0])
                wordnum_1 += 1
                break 
            elif obs_map_r[emission_1_rand_index].split('_')[1][0] != 'E' and int(obs_map_r[emission_1_rand_index].split('_')[1][0])+ syllable_num_1 == 10:                 
                emission_1.append(emission_1_rand_index)
                syllable_num_1 += int(obs_map_r[emission_1_rand_index].split('_')[1][0])
                wordnum_1 += 1
                break 
            else:
                emission_1_index = []
                for i in range(self.D):
                    
                    if obs_map_r[i].split('_')[1][0] != 'E' and int(obs_map_r[i].split('_')[1][0]) + syllable_num_1 <= 10:
                        emission_1_index.append(i)
                
                O_mat_1 = [self.O[states_1[wordnum_1]][i] for i in emission_1_index]
                emission_1_rand_index = random.choices(emission_1_index, O_mat_1)[0]
                   
                
                       
                if syllable_num_1 + int(obs_map_r[emission_1_rand_index].split('_')[1][0]) == 10:
                    emission_1.append(emission_1_rand_index)
                    syllable_num_1 += int(obs_map_r[emission_1_rand_index].split('_')[1][0])
                    wordnum_1 += 1
                    break
                
                elif syllable_num_1 + int(obs_map_r[emission_1_rand_index].split('_')[1][0]) < 10:
                    emission_1.append(emission_1_rand_index)
                    syllable_num_1 += int(obs_map_r[emission_1_rand_index].split('_')[1][0])
                    wordnum_1 += 1
                    continue 
                
        while syllable_num_2 < M:
            
            #print(emission)
            #print(states)
            states_2.append(random.choices(states_2_index,self.A[states_2[wordnum_2-1]])[0])
            emission_2_rand_index= random.choices(emission_2_full_index, self.O[states_2[wordnum_2]])[0]

            if obs_map_r[emission_2_rand_index].split('_')[1][0] == 'E'and int(obs_map_r[emission_2_rand_index].split('_')[1][1]) + syllable_num_2 == 10:                
                emission_2.append(emission_2_rand_index)
                syllable_num_2 += int(obs_map_r[emission_2_rand_index].split('_')[1][0])
                wordnum_2 += 1
                break 
            elif obs_map_r[emission_2_rand_index].split('_')[1][0] != 'E' and int(obs_map_r[emission_2_rand_index].split('_')[1][0])+ syllable_num_2 == 10:                 
                emission_2.append(emission_2_rand_index)
                syllable_num_2 += int(obs_map_r[emission_2_rand_index].split('_')[1][0])
                wordnum_2 += 1
                break 
            else:
                emission_2_index = []
                for i in range(self.D):
                    
                    if obs_map_r[i].split('_')[1][0] != 'E' and int(obs_map_r[i].split('_')[1][0]) + syllable_num_2 <= 10:
                        emission_2_index.append(i)
                
                O_mat_2 = [self.O[states_2[wordnum_2]][i] for i in emission_2_index]
                emission_2_rand_index = random.choices(emission_2_index, O_mat_2)[0]
                   
                
                       
                if syllable_num_2 + int(obs_map_r[emission_2_rand_index].split('_')[1][0]) == 10:
                    emission_2.append(emission_2_rand_index)
                    syllable_num_2 += int(obs_map_r[emission_2_rand_index].split('_')[1][0])
                    wordnum_2 += 1
                    break
                
                elif syllable_num_2 + int(obs_map_r[emission_2_rand_index].split('_')[1][0]) < 10:
                    emission_2.append(emission_2_rand_index)
                    syllable_num_2 += int(obs_map_r[emission_2_rand_index].split('_')[1][0])
                    wordnum_2 += 1
                    continue 
       
 
            

        #print(syllable_num)
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

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
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

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

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
