import tensorflow as tf
import pickle
import os
import string
import re
from numpy import array, argmax, random, take
from sklearn.model_selection import train_test_split
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import matplotlib.pyplot as plt

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        returned_data = pickle.load(file)
        file.close()
        return returned_data

# build NMT model
def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps,units):
      model = Sequential()
      model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
      model.add(LSTM(units))
      model.add(RepeatVector(out_timesteps))
      model.add(LSTM(units, return_sequences=True))
      model.add(Dense(out_vocab, activation='softmax'))
      return model

def dump_pickle(file_path, data, file_val):
    with open(file_path+'/'+str(file_val)+'.pickle', 'wb') as f:
        pickle.dump(data, f)
        f.close()


if __name__=="__main__":
    cur_dir = 'tokenized/'
    eng_encoded_seq = load_pickle(cur_dir+'English/1.pickle')
    port_encoded_seq = load_pickle(cur_dir + 'Portuguese/1.pickle')
    print(eng_encoded_seq)
    print("English: "+str(len(eng_encoded_seq[0])))
    print(port_encoded_seq)
    print("Protuguese: "+str(len(port_encoded_seq[0])))

    #devide 80:20 as train test from X(English)
    train_eng_enc_seq  = eng_encoded_seq[0:3200]
    test_eng_enc_seq = eng_encoded_seq[3200:]

    #devide 80:20 as train test from Y(Portuguese)
    train_port_enc_seq = port_encoded_seq[0:3200]
    test_port_enc_seq = port_encoded_seq[3200:]
    '''
    eng_vocab_size =  2437 #should take as argument or pickle
    port_vocab_size = 3183 #should take as argument or pickle
    max_eng_sen_word_length = 15 #should take as argument or pickle
    max_port_sen_word_length = 15 #should take as argument or pickle
    model = define_model(eng_vocab_size, port_vocab_size, max_eng_sen_word_length, max_port_sen_word_length, 512)
    rms = optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

    file_name = 'model.rnn.train'
    checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # train model
    history = model.fit(train_eng_enc_seq, train_port_enc_seq, epochs=30, batch_size=512, validation_split=0.2, callbacks = [checkpoint], verbose=1)

    #plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'validation'])
    plt.show()
    '''
    #dump test
    dump_pickle(cur_dir+'dump', test_eng_enc_seq, 'test_eng_enc_seq')
    dump_pickle(cur_dir+'dump', test_port_enc_seq, 'test_port_enc_seq')
