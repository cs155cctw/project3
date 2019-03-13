import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Lambda, Activation
from RNN_utility import poem_to_id, RNN_data_generator,poem_composer


corpus, dict = poem_to_id('data/shakespeare3.txt')
dict_len = len(dict)


# fix sequence length to 40 characters as required
seq_size = 40
# take sequence samples every 3 characters
skip_step = 3
batch_size = 1000
# generate sample x and y from shakespeare's poems
x, y = RNN_data_generator(corpus, seq_size, batch_size, dict_len, skip_step)


# Model created with constraints in the problem
model = Sequential()

# hidden_size within range of 100-200
hidden_size = 150

# a single layer of 150 LSTM units are applied as instructed
model.add(LSTM(hidden_size,input_shape=(seq_size, dict_len)))

# modify the temperature to obtain different probabiliy distribution
temp = 1.5
model.add(Lambda(lambda x: x / temp))
# a fully connected output layer with softmax nonlinearity
model.add(Dense(dict_len, activation = 'softmax'))

# Print a summary of the layers and weights in model
model.summary()

# with one-hot encoding the labels, use 'categorical_crossentropy' as the loss
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['categorical_accuracy'])

fit = model.fit(x, y, batch_size=30, epochs=10, verbose=1)

# generate random sentense seed of 40 charaters
# should change to the seed required 
batch_size = 2
skip_step = random.randint(0, len(corpus)-seq_size-1)
x, y = RNN_data_generator(corpus, seq_size, batch_size, dict_len, skip_step)
seed_sen = np.zeros((1, seq_size, dict_len))
seed_sen[0,:,:] = x[1,:,:]
poem_len = 200
pseudo_poem = poem_composer(model, seed_sen, poem_len, seq_size, dict, temp)

print(pseudo_poem)
