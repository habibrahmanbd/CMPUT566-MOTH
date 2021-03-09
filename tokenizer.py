##################################################################
#Tokenizer.py                                                    #
#   - Read Datasets from .txt files                              #
#   - splits the lines                                           #
#   - Apply tf-idf                                               #
#   - dump as matrix                                             #
##################################################################

import tensorflow as tf
import hickle as pickle
import os

def read_dataset(file_path):
    #Open from .txt files
    dataset = []
    with open(file_path) as f:
        dataset = f.readlines()
        f.close()
    return dataset

def split_input_target(dataset):
    datasetLength = len(dataset)

    # Split into English Sentence and Portuguese Sentences
    x =  [] #English
    y =  [] #Portuguese

    for line in dataset:
        splited = line.split('|')
        x.append(splited[0])
        y.append(splited[1])

    return [x, y]

def tokenize_tfidf(sentence_list):
    tok = tf.keras.preprocessing.text.Tokenizer()
    tok.fit_on_texts(sentence_list)
    return tok.sequences_to_matrix(tok.texts_to_sequences(sentence_list), mode='tfidf')

def dump_pickle(file_path, data, file_val):
    with open(file_path+'/'+str(file_val)+'.pickle', 'wb') as f:
        pickle.dump(data, f)
        f.close()

for i in range(1, 4):
    dataset = read_dataset('datasets/modified_datasets/dataset_'+str(i)+'.txt')
    x, y = split_input_target(dataset)

    print("----------TEST PRINT-----------")
    print("English: " + x[0])
    print("Portuguese: " + y[0])
    print("----------END TEST-------------")

    window = int(len(x) / 100)
    for j in range(1, 101):
        x_sequence_matrix = []
        y_sequence_matrix = []
        if j==100:
            x_sequence_matrix = tokenize_tfidf(x[(window*(j-1)):])
            y_sequence_matrix = tokenize_tfidf(y[(window*(j-1)):])
        else:    
            x_sequence_matrix = tokenize_tfidf(x[(window*(j-1)):(window*j)])
            y_sequence_matrix = tokenize_tfidf(y[(window*(j-1)):(window*j)])

        print("----------TEST X and Y---------")
        print("--------------X----------------")
        print(len(x_sequence_matrix[0]))
        print("--------------Y----------------")
        print(len(y_sequence_matrix[0]))
        print("----------END X and Y----------")

        directory1 = 'tokenized/modified_dataset/data'+str(i)+'/English'
        if not os.path.exists(directory1):
            os.makedirs(directory1, exist_ok=True)

        directory2 = 'tokenized/modified_dataset/data'+str(i)+'/Portuguese'
        if not os.path.exists(directory2):
            os.makedirs(directory2, exist_ok=True)

        print("---------DUMP X As Pickle------")
        dump_pickle(directory1, x_sequence_matrix, j)
        print("---------DUMP X END------------")

        print("---------DUMP Y As Pickle------")
        dump_pickle(directory2, y_sequence_matrix, j)
        print("---------DUMP Y END------------")

