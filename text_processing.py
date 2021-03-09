##################################################################
#text_processing.py                                              #
#   - Read Datasets from .txt files                              #
#   - splits the lines                                           #
#   - Apply tf-idf                                               #
#   - dump as matrix                                             #
##################################################################

import tensorflow as tf
import pickle
import os
import string
import re
from numpy import array, argmax, random, take
import pandas as pd
import keras
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 200)

def read_dataset(file_path):
    #Open from .txt files
    dataset = []
    with open(file_path, encoding='utf-8') as f:
        dataset = f.readlines()
        f.close()
    return dataset

def split_input_target(dataset):
    datasetLength = len(dataset)

    # Split into English Sentence and Portuguese Sentences
    eng_sen =  [] #English Sentence
    port_sen =  [] #Portuguese Sentence

    for line in dataset:
        splited = line.split('|')
        eng_sen.append(splited[0])
        port_sen.append(splited[1])

    return [eng_sen, port_sen]

def cleaning_punctuation_and_uppercase(sentence_list):
    sentence_list  = [(sen.translate(str.maketrans('', '', string.punctuation))).lower() for sen in sentence_list]
    return sentence_list

def visualize_length_of_sentences(senX, senY):
    senX = [len(sen.split()) for sen in senX]
    senY = [len(sen.split()) for sen in senY]
    length_df = pd.DataFrame({'English': senX, 'Portuguese': senY})
    length_df.hist(bins = 30)
    plt.show()

def tokenizer(sentence_list):
    tok = tf.keras.preprocessing.text.Tokenizer()
    tok.fit_on_texts(sentence_list)
    return  tok #tok.sequences_to_matrix(tok.texts_to_sequences(sentence_list), mode='tfidf')

# Text Encoding into sequences and pad to make equal feature length to Train NN
def encode_text_to_sequences(tokenizer, max_sen_length, sentence_list):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(sentence_list)
    # pad sequences with 0 values
    seq = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_sen_length, padding='post')
    return seq

def dump_pickle(file_path, data, file_val):
    with open(file_path+'/'+str(file_val)+'.pickle', 'wb') as f:
        pickle.dump(data, f)
        f.close()

#from google.colab import drive
#drive.mount('/content/gdrive')


for i in range(1, 4):
    
    dataset = read_dataset('datasets/modified_datasets/dataset_'+str(i)+'.txt')
    eng_sen, port_sen = split_input_target(dataset)

    #Cleaning
    eng_sen = cleaning_punctuation_and_uppercase(eng_sen)
    port_sen = cleaning_punctuation_and_uppercase(port_sen)

    #Plot Sentences
    #visualize_length_of_sentences(eng_sen, port_sen)

    #tokenize
    eng_tok = tokenizer(eng_sen)
    port_tok = tokenizer(port_sen)

    #Max word length in Sentence
    max_eng_sen_word_length  = 6
    max_port_sen_word_length = 7

    #Vocab Size
    eng_vocab_size = len(eng_tok.word_index)+1
    port_vocab_size = len(port_tok.word_index)+1

    #encoding text to sequence
    eng_encoded_seq = encode_text_to_sequences(eng_tok, max_eng_sen_word_length, eng_sen)
    port_encoded_seq = encode_text_to_sequences(port_tok, max_port_sen_word_length, port_sen)

    #print
    print(eng_encoded_seq)
    print(port_encoded_seq)

    #dump as pickle
    directory_eng = os.path.dirname('tokenized/English/')
    directory_port = os.path.dirname('tokenized/Portuguese/')
    if not os.path.exists(directory_eng):
        os.makedirs(directory_eng)
    if not os.path.exists(directory_port):
        os.makedirs(directory_port)
    dump_pickle(directory_eng, eng_encoded_seq, i)
    dump_pickle(directory_port, port_encoded_seq, i)

