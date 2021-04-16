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

for i in range(1, 4):

    with open("tokenized/Portuguese/predict.d"+str(i)+".final.txt", "r") as f:
        data = f.read()
        print(len(data))
        f.close()


    #data = data.replace('[', '')
    #data = data.replace(']', '')
    #data = data.replace('\'', '')

    #data = data.split(',')


    test_data = read_dataset('datasets/testing_datasets/test.txt')

    test_data_eng, test_data_port = split_input_target(test_data)

    #test_data_eng = cleaning_punctuation_and_uppercase(test_data_eng)

    print(len(test_data_eng))
    print(len(data))

    with open("datasets/RNN_Result/predict.d"+str(i)+".gold_format.txt", "w") as f:
        for i in range(len(data)):
            print(test_data_eng[i]+'|'+data[i], file=f)
        f.close()
