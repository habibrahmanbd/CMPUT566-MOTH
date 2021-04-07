#######################################################################
# Content:
# Functions for processing the raw data.
#######################################################################

import numpy as np
import pandas as pd
import string
from random import choices


#######################################################################
# Functions:

# Creates one of the three modified datasets.
# dataset_path: the directory where the raw data is.
# mode: determines which dataset is to be created
#       mode = 1: dateset only has best translation
#       mode = 2: dateset has each translation only once
#       mode = 3: dateset has each translation appear based on weight
def create_modified_dataset(dataset_path, mode):
    with open(dataset_path,'r',encoding="utf-8") as f:
        lines = np.loadtxt(f,delimiter='|', dtype='U')
    
    max_length = 0
    for line in lines:
        for sentence in line:
            if len(sentence) > max_length:
                max_length = len(sentence)
    
    modified_dataset = np.empty((1000000,2),dtype='U'+str(max_length+1))

    if mode == 1:
        i = 0
        j = 0
        for line in lines:
            if line[0].startswith('prompt_'):
                modified_dataset[j,0] = line[1]
                modified_dataset[j,1] = lines[i+1,0]
                j += 1
            i += 1

    elif mode == 2:
        j = 0
        for line in lines:
            if line[0].startswith('prompt_'):
                promt = line[1]
            else:
                modified_dataset[j,0] = promt
                modified_dataset[j,1] = line[0]
                j += 1

    elif mode == 3:
        length = lines.shape[0]
        i = 0
        j = 0
        while i < length:
            promt = lines[i,1]
            i += 1

            translations = np.empty((0),dtype='U')
            weights = np.empty((0))

            while i < length and not(lines[i,0].startswith('prompt_')):
                translations = np.append(translations,lines[i,0])
                weights = np.append(weights,float(lines[i,1]))
                i += 1

            for _ in range(130):
                modified_dataset[j,0] = promt
                modified_dataset[j,1] = choices(translations,weights)[0]
                j += 1

    else:
        print("Error: Not a valid mode value.")
    
    return np.delete(modified_dataset,(modified_dataset == '')[:,0],0)


# Creates dev or test datasets.
# dataset_path: the directory where the raw data is.
def create_testing_dataset(dataset_path):
    with open(dataset_path,'r',encoding="utf-8") as f:
        lines = pd.read_table(f,delimiter='|', dtype='U',header=None)
    
    modified_dataset = pd.DataFrame(None,index=range(lines.shape[0]),columns=['promt','translation','weights'], dtype='U' )

    i = 0
    j = 0
    for line in lines.itertuples(index=False,name=None):
        if line[0].startswith('prompt_'):
            promt = line[1]
        else:
            modified_dataset.iat[j,0] = promt
            modified_dataset.iat[j,1] = line[0]
            modified_dataset.iat[j,2] = line[1]
            j += 1
        i += 1

    
    return modified_dataset.dropna(0)




# Creates Amazon's Answers.
# dataset_path: the directory where the raw data is.
def create_amazon_baseline(dataset_path):
    with open(dataset_path,'r',encoding="utf-8") as f:
        lines = pd.read_table(f,delimiter='|', dtype='U',header=None)
    
    modified_dataset = pd.DataFrame('__',index=range(500),columns=['promt','translation'], dtype='U' )

    i = 0
    j = 0
    for line in lines.itertuples(index=False,name=None):
        if line[0].startswith('prompt_'):
            modified_dataset.iat[j,0] = line[1]
            modified_dataset.iat[j,1] = lines.iat[i+1,0]
            j += 1
        i += 1

    
    return modified_dataset

# Creates Worst Translation Baseline Answers.
# dataset_path: the directory where the raw data is.
def create_worst_baseline(dataset_path):
    with open(dataset_path,'r',encoding="utf-8") as f:
        lines = pd.read_table(f,delimiter='|', dtype='U',header=None)
    
    modified_dataset = pd.DataFrame('__',index=range(500),columns=['promt','translation'], dtype='U' )

    max_length = lines.shape[0]-1

    i = 0
    j = 0
    for line in lines.itertuples(index=False,name=None):
        if line[0].startswith('prompt_'):
            promt = line[1]
        elif i == max_length or lines.iat[i+1,0].startswith('prompt_'):
            modified_dataset.iat[j,0] = promt
            modified_dataset.iat[j,1] = lines.iat[i,0]
            j += 1
        i += 1

    
    return modified_dataset


def cleaning_punctuation_and_uppercase(sentence):
    sentence  = (sentence.translate(str.maketrans('', '', string.punctuation))).lower().strip()
    return sentence

# Convert dataset to gold format.
# dataset_path: the directory where the raw data is.
# reference_path: the directory of the dataset the raw data is trying to predict.
# head: does raw data have a header line (None for no header and 0 for a header).
def convert_to_gold(dataset_path,reference_path,head):
    with open(dataset_path,'r',encoding="utf-8") as f:
        dataset_data = pd.read_table(f,delimiter='|', dtype='U',header=head)

    with open(reference_path,'r',encoding="utf-8") as f:
        reference_data = pd.read_table(f,delimiter='|', dtype='U',header=None)
    
    gold_dataset = pd.DataFrame('',index=range(501*3), columns=['line'], dtype='U')

                
    portu = ''
    promt_found = False
    promt_id = ''
    promt = ''
    i = 0
    for reference in reference_data.itertuples(index=False,name=None):
        if reference[0].startswith('prompt_'):
            if promt_found:
                gold_dataset.iat[i,0] = portu.strip()
                gold_dataset.iat[i+1,0] = ''
                i += 2
                promt_found = False
            else:
                gold_dataset.iat[i,0] = promt_id + '|' + promt
                gold_dataset.iat[i+1,0] = ''
                gold_dataset.iat[i+2,0] = ''
                i += 3

            for data in dataset_data.itertuples(index=False,name=None):
                if reference[1] == data[0]:
                    gold_dataset.iat[i,0] = reference[0] + '|' + data[0]
                    i += 1
                    portu = data[1]
                    promt_found = True
                    break

            promt_id = reference[0]
            promt = reference[1]
        else:
            if cleaning_punctuation_and_uppercase(portu) == cleaning_punctuation_and_uppercase(reference[0]):
                weight = reference[1]
    
    gold_dataset.iat[i,0] = portu
    gold_dataset.iat[i+1,0] = ''


    
    return gold_dataset.drop([0,1,2])


#######################################################################
# Test code

# output = convert_to_gold('CMPUT566-MOTH/datasets/Transformer_Result/result_dataset_1_trial1.csv','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# print(output)
# print(output.shape)

# print("End")




# Save Modified Datasets

# output = create_modified_dataset('CMPUT566-MOTH/datasets/staple-2020/en_pt/train.en_pt.2020-01-13.gold.txt', 1)

# np.savetxt('CMPUT566-MOTH/datasets/modified_datasets/dataset_1.txt', output, fmt='%s',delimiter='|',encoding='utf-8')

# print("1 done")

# output = create_modified_dataset('CMPUT566-MOTH/datasets/staple-2020/en_pt/train.en_pt.2020-01-13.gold.txt', 2)

# np.savetxt('CMPUT566-MOTH/datasets/modified_datasets/dataset_2.txt', output, fmt='%s',delimiter='|',encoding='utf-8')

# print("2 done")

# output = create_modified_dataset('CMPUT566-MOTH/datasets/staple-2020/en_pt/train.en_pt.2020-01-13.gold.txt', 3)

# np.savetxt('CMPUT566-MOTH/datasets/modified_datasets/dataset_3.txt', output, fmt='%s',delimiter='|',encoding='utf-8')

# print("3 done")

# print("End")


# Save Test and Dev Datasets

# output = create_testing_dataset('CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt')

# output.to_csv('CMPUT566-MOTH/datasets/testing_datasets/test.txt',sep="|",encoding='utf-8',index=False,header=False)

# output = create_testing_dataset('CMPUT566-MOTH/datasets/staple-2020/en_pt/dev.en_pt.2020-02-20.gold.txt')

# output.to_csv('CMPUT566-MOTH/datasets/testing_datasets/dev.txt',sep="|",encoding='utf-8',index=False,header=False)


# Save Amazon Basline Dataset

# output = create_amazon_baseline('CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.aws_baseline.pred.txt')

# output.to_csv('CMPUT566-MOTH/datasets/baseline_datasets/amazon.txt',sep="|",encoding='utf-8',index=False,header=False)



# Save Worst Basline Dataset

# output = create_worst_baseline('CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt')

# output.to_csv('CMPUT566-MOTH/datasets/baseline_datasets/worst.txt',sep="|",encoding='utf-8',index=False,header=False)




# Save Gold version of the Transformer's Predictions Datasets

# output = convert_to_gold('CMPUT566-MOTH/datasets/Transformer_Result/result_dataset_1_trial1.csv','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_transformer/dataset1_trial1.txt', output, fmt='%s',encoding='utf-8')

# output = convert_to_gold('CMPUT566-MOTH/datasets/Transformer_Result/result_dataset_1_trial2.csv','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_transformer/dataset1_trial2.txt', output, fmt='%s',encoding='utf-8')

# output = convert_to_gold('CMPUT566-MOTH/datasets/Transformer_Result/result_dataset_1_trial3.csv','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_transformer/dataset1_trial3.txt', output, fmt='%s',encoding='utf-8')


# output = convert_to_gold('CMPUT566-MOTH/datasets/Transformer_Result/result_dataset_2_trial1.csv','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_transformer/dataset2_trial1.txt', output, fmt='%s',encoding='utf-8')

# output = convert_to_gold('CMPUT566-MOTH/datasets/Transformer_Result/result_dataset_2_trial2.csv','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_transformer/dataset2_trial2.txt', output, fmt='%s',encoding='utf-8')

# output = convert_to_gold('CMPUT566-MOTH/datasets/Transformer_Result/result_dataset_2_trial3.csv','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_transformer/dataset2_trial3.txt', output, fmt='%s',encoding='utf-8')


# output = convert_to_gold('CMPUT566-MOTH/datasets/Transformer_Result/result_dataset_3_trial1.csv','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_transformer/dataset3_trial1.txt', output, fmt='%s',encoding='utf-8')

# output = convert_to_gold('CMPUT566-MOTH/datasets/Transformer_Result/result_dataset_3_trial2.csv','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_transformer/dataset3_trial2.txt', output, fmt='%s',encoding='utf-8')

# output = convert_to_gold('CMPUT566-MOTH/datasets/Transformer_Result/result_dataset_3_trial3.csv','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_transformer/dataset3_trial3.txt', output, fmt='%s',encoding='utf-8')




# Save Gold version of the RNN's Predictions Datasets

# output = convert_to_gold('CMPUT566-MOTH/datasets/RNN_Result/predict.habib1.updated.txt','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_rnn/dataset1_trial1_h.txt', output, fmt='%s',encoding='utf-8')

# output = convert_to_gold('CMPUT566-MOTH/datasets/RNN_Result/predict.habib2.updated.txt','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_rnn/dataset2_trial1_h.txt', output, fmt='%s',encoding='utf-8')


# output = convert_to_gold('CMPUT566-MOTH/datasets/RNN_Result/predict.maisha1.updated.txt','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_rnn/dataset1_trial1_m.txt', output, fmt='%s',encoding='utf-8')

# output = convert_to_gold('CMPUT566-MOTH/datasets/RNN_Result/predict.maisha2.updated.txt','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_rnn/dataset2_trial1_m.txt', output, fmt='%s',encoding='utf-8')

# output = convert_to_gold('CMPUT566-MOTH/datasets/RNN_Result/predict.maisha3.updated.txt','CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt',0)

# np.savetxt('CMPUT566-MOTH/datasets/gold_rnn/dataset3_trial1_m.txt', output, fmt='%s',encoding='utf-8')
