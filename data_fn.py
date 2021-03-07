#######################################################################
# Content:
# Functions for processing the raw data.
#######################################################################

import numpy as np
from random import choices

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




# Test code

# output = create_modified_dataset('CMPUT566-MOTH/datasets/staple-2020/en_pt/train.en_pt.2020-01-13.gold.txt', 3)

# print(output[0:100])
# print(output.shape)

# print("End")




# Save Datasets

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