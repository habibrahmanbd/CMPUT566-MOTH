## CMPUT566-MOTH

### Requirements:
 - tensorflow
### Directory Structure
.
├── data_fn.py
├── datasets
│   ├── modified_datasets
│   │   ├── dataset_1.txt
│   │   ├── dataset_2.txt
│   │   └── dataset_3.txt
│   ├── staple-2020
│   │   ├── en_pt
│   │   │   ├── dev.en_pt.2020-02-20.gold.txt
│   │   │   ├── dev.en_pt.aws_baseline.pred.txt
│   │   │   ├── test.en_pt.2020-02-20.gold.txt
│   │   │   ├── test.en_pt.aws_baseline.pred.txt
│   │   │   ├── train.en_pt.2020-01-13.gold.txt
│   │   │   └── train.en_pt.aws_baseline.pred.txt
│   │   └── README.txt
│   └── staple-2020-full-data.tar.gz
├── README.md
├── tokenized
│   └── modified_dataset
│       └── data1
│           ├── English
│           └── Portuguese
└── tokenizer.py

### Data Argumentation 1
```
├── tokenized
│   └── modified_dataset
│       └── data1
│           ├── English
│           └── Portuguese
└── tokenizer.py
```

 - `tokenizer.py`: Dumps the pickle file for English and Portuguese Sentences
 - `English`: Each row of this file is tfidf data for a single English Sentence
 - `Portuguese`: Each row of this file is tfidf data for the translated sentence
 **English to Portuguese maps with row by row**
