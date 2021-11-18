## Data Augmentation in Neural Machine Translation: A Case-study for English to Portuguese Translation
We have tested several data augmentation for better training for this project, and hence better translation results. We have translated English sentences into Portuguese. The input was a single English sentence, and the output was a single Portuguese sentence that translated the English sentence.

### Graduate Students
 - [Habibur Rahman](https://habibrahman.me)
 - [Omar Al-Shamali]()
 - [Md. Tanvir Alam Anik]()
 - [Maisha Binte Moin]()
### Dependencies
 - tensorflow
 - keras
 - GPU
 - Linux Machine

 ````shell
 # Quick Installation steps
 pip3 install -r requirements.txt
 ````

### Directory Structure
```
.
├── baselines.py                                                                 # Baseline Result Calculation
├── bleu_score.py                                                                # Script to calculate Blue Score
├── data_fn.py                                                                   # Script to Convert Pred. Text in Gold Format
├── datasets                                                                     # Contains the Project Dataset and Result
│   ├── baseline_datasets                                                        # Dataset for Baseline models
│   │   ├── amazon.txt                                                           # Baseline Data of Amazon
│   │   └── worst.txt                                                            # Baseline Data of Worst
│   ├── F1_Score                                                                 # Folder for print F1 Score
│   │   └── results_RNN.txt
│   ├── gold_rnn                                                                 # Gold Format Prediction of RNN
│   │   ├── dataset1_m.txt
│   │   ├── dataset1_trial1_h.txt
│   ├── gold_transformer                                                         # Gold Format Prediction of Transformer
│   │   ├── dataset1_trial1.txt
│   │   ├── dataset3_trial3.txt
│   │   └── test.txt                                                             # Test for Transformer
│   ├── modified_datasets                                                        # Modified Dataset
│   │   ├── dataset_1.txt                                                        # Data Argumentation 1
│   │   ├── dataset_2.txt                                                        # Data Argumentation 2
│   │   └── dataset_3.txt                                                        # Data Argumentation 3
│   ├── RNN_Result                                                               # RNN Results
│   │   ├── dev_best
│   │   ├── predict.habib1.txt
│   │   ├── predict.habib1.updated.txt
│   │   ├── predict.maisha3.gold_format.txt
│   │   └── ...
│   ├── staple-2020                                                              # Staple 2020 Original Dataset for en_pt
│   │   ├── en_pt
│   │   │   ├── dev.en_pt.2020-02-20.gold.txt
│   │   │   └── ...
│   │   └── README.txt
│   ├── testing_datasets                                                         # Dataset for Testing and Validation
│   │   ├── dev_best.txt
│   │   ├── dev.txt
│   │   └── test.txt
│   └── Transformer_Result                                                       # Transformer Result
│       ├── result_dataset_1_trial1.csv
│       └── ...
├── images                                                                       # Necessary Images of this Projects
│   ├── Figure_1.png
├── RNN.sh                                                                       # Run RNN to Generate Final Result
├── text_processing.py                                                           # Text Processing, Model Training, Output Prediction, etc
├── tokenized                                                                    # Dump Tokenize Data, Results
│   ├── dump
│   │   ├── test_eng_enc_seq.pickle
│   │   └── test_port_enc_seq.pickle
│   ├── English
│   │   ├── eng_tok1.pickle
│   └── Portuguese
│       ├── 1.pickle
│       ├── ...
├── tokenizer.py
├── transformer                                                                          # Folder For Transformer
│   ├── CMPUT566_Eng_Por_Translation_Transformer_Model_dataset123.ipynb
│   └── Restore_checkpoints_dataset123.ipynb
```
### Instructions

### Model with RNN

#### Step 1: Create Modified Datasets
```
python3 create_modified_datasets.py
```
#### Step 2: Run Script for RNN Result in Bleu Score
```
./RNN.sh
```
#### Step 3: Run Command for Weighted F1 Macro of RNN

```
git clone https://github.com/duolingo/duolingo-sharedtask-2020.git
cd duolingo-sharedtask-2020
```
*Weighted Score for Dataset 1*
```
python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt  --predfile ../CMPUT566-MOTH/datasets/gold_rnn/dataset1.txt
```
*Weighted Score for Dataset 2*
```
python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt  --predfile ../CMPUT566-MOTH/datasets/gold_rnn/dataset2.txt
```
*Weighted Score for Dataset 3*
```
python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/staple-2020/en_pt/test.en_pt.2020-02-20.gold.txt  --predfile ../CMPUT566-MOTH/datasets/gold_rnn/dataset3.txt
```

### Model with Transformer

#### Step 1: Create Modified Datasets
```
python3 create_modified_datasets.py
```
#### Step 2: Create Training Checkpoints
```
python3 cmput566_eng_por_translation_transformer_model_dataset123.py
```
#### Step 3: Create Dataset for Bleu & F1 score
```
python3 restore_checkpoints_dataset123.py
````
#### Step 4: Run Command for Weighted F1 Macro of RNN

```
git clone https://github.com/duolingo/duolingo-sharedtask-2020.git
cd duolingo-sharedtask-2020
```
*Weighted Score for Dataset 1*
```
python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/gold_transformer/test.txt  --predfile ../CMPUT566-MOTH/datasets/gold_transformer/dataset1_trial1.txt

python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/gold_transformer/test.txt  --predfile ../CMPUT566-MOTH/datasets/gold_transformer/dataset1_trial2.txt

python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/gold_transformer/test.txt  --predfile ../CMPUT566-MOTH/datasets/gold_transformer/dataset1_trial3.txt
```
*Weighted Score for Dataset 2*
```
python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/gold_transformer/test.txt  --predfile ../CMPUT566-MOTH/datasets/gold_transformer/dataset2_trial1.txt

python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/gold_transformer/test.txt  --predfile ../CMPUT566-MOTH/datasets/gold_transformer/dataset2_trial2.txt

python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/gold_transformer/test.txt  --predfile ../CMPUT566-MOTH/datasets/gold_transformer/dataset2_trial3.txt
```
*Weighted Score for Dataset 3*
```
python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/gold_transformer/test.txt  --predfile ../CMPUT566-MOTH/datasets/gold_transformer/dataset3_trial1.txt

python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/gold_transformer/test.txt  --predfile ../CMPUT566-MOTH/datasets/gold_transformer/dataset3_trial2.txt

python3 staple_2020_scorer.py --goldfile ../CMPUT566-MOTH/datasets/gold_transformer/test.txt  --predfile ../CMPUT566-MOTH/datasets/gold_transformer/dataset3_trial3.txt
```

#### Transformer Codes are available to run in Google Colab, Code: https://drive.google.com/drive/folders/1nv0kY3KEnn3eh_SJok21bZE9CLZF9C_E?usp=sharing
