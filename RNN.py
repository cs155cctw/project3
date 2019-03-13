import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Lambda
from RNN_utility import poem_to_id, sentence_to_id, RNN_data_generator, poem_composer


corpus, dict = poem_to_id('data/shakespeare4.txt')
dict_len = len(dict)


# fix sequence length to 40 characters as required
seq_size = 40
# take sequence samples every 3 characters
skip_step = 3
batch_size = int(round((len(corpus)-seq_size)*1.0/skip_step))
print('batch_size = ', batch_size)
# generate sample x and y from shakespeare's poems
x, y = RNN_data_generator(corpus, seq_size, batch_size, dict_len, skip_step)

# hidden_size within range of 100-200
hidden_size = 150

# modify the temperature to obtain different probabiliy distribution
temp_list = [0.25, 0.75, 1.5]
for temp in temp_list:
    # Model created with constraints in the problem
    model = Sequential()
    # a single layer of 150 LSTM units are applied as instructed
    model.add(LSTM(hidden_size,input_shape=(seq_size, dict_len)))

    model.add(Lambda(lambda x: x / temp))
    # a fully connected output layer with softmax nonlinearity
    
    model.add(Dense(dict_len, activation = 'softmax'))

    # Print a summary of the layers and weights in model
    model.summary()

    # with one-hot encoding the labels, use 'categorical_crossentropy' as the loss
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['categorical_accuracy'])
    
    # change epochs to converge the accuracy
    fit = model.fit(x, y, batch_size=30, epochs=40, verbose=1)

    model_file = 'model_t'+str(temp)+'.h5'
    model.save_weights(model_file)

    # seed sentence
    seed = "shall i compare thee to a summer's day?\n"
    seed_id = sentence_to_id(seed, dict)
    seed_sen = np.zeros((1, seq_size, dict_len))
    seed_sen[0,:,:] = seed_id
    poem_len = 800
    pseudo_poem = poem_composer(model, seed_sen, poem_len, seq_size, dict, temp)
    print(pseudo_poem)

## reload files
#for temp in temp_list:
#    model = Sequential()
#    model.add(LSTM(hidden_size,input_shape=(seq_size, dict_len)))
#    model.add(Lambda(lambda x: x / temp))
#    model.add(Dense(dict_len, activation = 'softmax'))
#    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['categorical_accuracy'])
#    model_file = 'model_t'+str(temp)+'.h5'
#    model.load_weights(model_file)
#    pseudo_poem = poem_composer(model, seed_sen, poem_len, seq_size, dict, temp)
#    print(pseudo_poem)
