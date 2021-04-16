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
    root = ''#/content/gdrive/MyDrive/ColabNotebooks/CMPUT566/'
    directory_eng = os.path.dirname(root+'tokenized/English/')
    directory_port = os.path.dirname(root+'tokenized/Portuguese/')
    for i in range(1, 4):
        print("------------------LOAD----------------")
        val_eng_enc_seq = load_pickle(directory_eng+'/validation_enc_seq'+str(i)+'.pickle')
        val_port_enc_seq = load_pickle(directory_port+'/validation_enc_seq'+str(i)+'.pickle')

#        test_eng_enc_seq = load_pickle(directory_eng+'/test_enc_seq'+str(i)+'.pickle')
#        test_port_enc_seq = load_pickle(directory_port+'/test_enc_seq'+str(i)+'.pickle')

        train_eng_enc_seq = load_pickle(directory_eng+'/train'+str(i)+'.pickle')
        train_port_enc_seq = load_pickle(directory_port+'/train'+str(i)+'.pickle')

        eng_vocab_size =  load_pickle(directory_eng+'/eng_vocab'+str(i)+'.pickle')
        port_vocab_size = load_pickle(directory_port+'/port_vocab'+str(i)+'.pickle')

        max_eng_sen_word_length = load_pickle(directory_eng+'/max_eng_sen_word_length'+str(i)+'.pickle')
        max_port_sen_word_length = load_pickle(directory_port+'/max_port_sen_word_length'+str(i)+'.pickle')

        print(max_eng_sen_word_length)
        print(max_port_sen_word_length)

        unit = 0
        if i == 1:
          epoch = 100
          unit = 512
        else:
          epoch = 30
          unit = 1024
        
        model = define_model(eng_vocab_size, port_vocab_size, max_eng_sen_word_length, max_port_sen_word_length, unit)
        adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=adam, loss='sparse_categorical_crossentropy')

        print(model.summary())

        file_name = root+'model.h1.d'+str(i)+'_11_apr_21'
        checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        # train model
        history = model.fit(train_eng_enc_seq, train_port_enc_seq, epochs=epoch, batch_size=64, validation_steps=None, validation_data = (val_eng_enc_seq, val_port_enc_seq), callbacks = [checkpoint], verbose=1)

        #plot
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['train', 'validation'])
        plt.show()
