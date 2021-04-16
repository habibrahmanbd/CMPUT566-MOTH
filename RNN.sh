#!/bin/bash
python3 text_processing.py
python3 model_train_rnn.py
python3 predict_rnn.py
python3 preprocess_before_promt.py
python3 bleu_score_rnn.py
python3 gold_style_pred_rnn.py
