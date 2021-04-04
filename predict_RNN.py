import keras
from keras.models import load_model
import pickle
import pandas as pd
import os

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

def dump_pickle(file_path, data, file_val):
    with open(file_path+'/'+str(file_val)+'.pickle', 'wb') as f:
        pickle.dump(data, f)
        f.close()


if __name__=="__main__":
    directory_eng = os.path.dirname('tokenized/English/')
    directory_port = os.path.dirname('tokenized/Portuguese/')
    print('------------------------------PREDICT-------------------------------')
    for i in range(1, 4):
        model = load('model.rnn.train'+str(i))

        eng_tok = load_pickle(directory_eng+'eng_tok'+str(i)+'.pickle')
        port_tok = load_pickle(directory_port+'port_tok'+str(i)+'.pickle')

        test_eng_enc_seq = load_pickle(directory_eng+'test_eng_enc_seq'+str(i)+'.pickle')
        test_port_enc_seq = load_pickle(directory_port+'test_port_enc_seq'+str(i)+'.pickle')

        preds = model.predict_classes(test_eng_enc_seq)

        pred_text = predicted_text(preds, port_tok)
        dump_pickle(directory_port , pred_text, 'predict'+str(i))
        actc_text = predicted_text(test_port_enc_seq, port_tok)

        pred_df = pd.DataFrame({'actual' : actc_text, 'predicted' : pred_text})
        pred_df.sample(15)
