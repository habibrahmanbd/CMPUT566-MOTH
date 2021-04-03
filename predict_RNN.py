import keras
from keras.models import load_model
import pickle
import pandas as pd


def load(model_name):
    return load_model(model_name)

def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None

def predicted_text(preds, out_tok):
    pred_text = []
    for i in preds:
        temp = []
        for j in range(len(i)):
            t = get_word(i[j], out_tok)
            if j > 0:
                if (t == get_word(i[j-1], out_tok) or t == None):
                    temp.append('')
                else:
                    temp.append(t)
            else:
                if t == None:
                    temp.append('')
                else:
                    temp.append(t)
        pred_text.append(' '.join(temp))
    return pred_text

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        returned_data = pickle.load(file)
        file.close()
        return returned_data

if __name__=="__main__":
    cur_dir = 'tokenized/'
    model = load('model.rnn.train')
    eng_tok = load_pickle('tokenized/English/eng_tok1.pickle')
    port_tok = load_pickle('tokenized/Portuguese/port_tok1.pickle')
    test_eng_enc_seq = load_pickle(cur_dir+'dump/test_eng_enc_seq.pickle')
    test_port_enc_seq = load_pickle(cur_dir+'dump/test_port_enc_seq.pickle')
    preds = model.predict_classes(test_eng_enc_seq)

    pred_text = predicted_text(preds, port_tok)
    actc_text = predicted_text(test_port_enc_seq, port_tok)

    pred_df = pd.DataFrame({'actual' : actc_text, 'predicted' : pred_text})
    pred_df.sample(15)
