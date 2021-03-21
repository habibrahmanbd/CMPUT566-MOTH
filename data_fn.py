#######################################################################
# Content:
# Functions for processing the raw data.
#######################################################################

import numpy as np
import pandas as pd
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

#######################################################################
# Test code

# output = create_testing_dataset('CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt')

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