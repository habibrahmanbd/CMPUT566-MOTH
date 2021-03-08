##################################################################
#Tokenizer.py                                                    #
#   - Read Datasets from .txt files                              #
#   - splits the lines                                           #
#   - Apply tf-idf                                               #
#   - dump as matrix                                             #
##################################################################

import tensorflow as tf
import pickle

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

def dump_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        f.close()

for i in range(1, 4):
    dataset = read_dataset('datasets/modified_datasets/dataset_'+str(i)+'.txt')
    x, y = split_input_target(dataset)

    print("----------TEST PRINT----------")
    print("English: " + x[0])
    print("Portuguese: " + y[0])
    print("----------END TEST------------")

    x_sequence_matrix = tokenize_tfidf(x)
    y_sequence_matrix = tokenize_tfidf(y)

    print("----------TEST X and Y---------")
    print("--------------X----------------")
    print(x_sequence_matrix)
    print("--------------Y----------------")
    print(y_sequence_matrix)
    print("----------END X and Y----------")

    print("---------DUMP X As Pickle------")
    dump_pickle('tokenized/modified_dataset/data'+str(i)+'/English', x_sequence_matrix)
    print("---------DUMP X END------------")

    print("---------DUMP Y As Pickle------")
    dump_pickle('tokenized/modified_dataset/data'+str(i)+'/Portuguese', y_sequence_matrix)
    print("---------DUMP Y END------------")

