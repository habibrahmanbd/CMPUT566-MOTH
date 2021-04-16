import keras
from keras.models import load_model
import pickle
import pandas as pd
import os
import sys

def load(model_name):
    return load_model(model_name)

def get_word(n, tokenizer):
    Ret = None
    try:
      Ret = tokenizer.index_word[n]
    except:
      Ret = None
    return Ret

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
        pred_text.append((' '.join(temp)).strip())
    return pred_text

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        returned_data = pickle.load(file)
        file.close()
        return returned_data

def dump_pickle(file_path, data, file_val):
    with open(file_path+'/'+str(file_val)+'.txt', "w", encoding="utf-8") as f:
        for row in data:
          print(row, file=f)
    f.close()

if __name__=="__main__":
    root = ''#ontent/gdrive/MyDrive/ColabNotebooks/CMPUT566/'
    directory_eng = root + os.path.dirname('tokenized/English/')
    directory_port = root + os.path.dirname('tokenized/Portuguese/')
    print('------------------------------PREDICT-------------------------------')
    for i in range(1, 4):
        model = load(root+'model.h1.d'+str(i)+'_11_apr_21')

        eng_tok = load_pickle(directory_eng+'/eng_tok'+str(i)+'.pickle')
        port_tok = load_pickle(directory_port+'/port_tok'+str(i)+'.pickle')

        test_eng_enc_seq = load_pickle(directory_eng+'/test_enc_seq'+str(i)+'.pickle')
        test_port_enc_seq = load_pickle(directory_port+'/test_enc_seq'+str(i)+'.pickle')
        
        print("Len of Test English: " + str(len(test_eng_enc_seq)))
        print("Len of Test Portuguese: " + str(len(test_port_enc_seq)))

        pred_text_final = []
        batch = len(test_eng_enc_seq) // 1024
        #print(len(test_eng_enc_seq))
        
        for j in range(1, batch+1):
          preds = model.predict_classes(test_eng_enc_seq[(j-1)*1024: j*1024])
          pred_text = predicted_text(preds, port_tok)
          for k in pred_text:
            pred_text_final.append(k)
        
        preds = model.predict_classes(test_eng_enc_seq[batch*1024:])
        pred_text = predicted_text(preds, port_tok)

        for k in pred_text:
          pred_text_final.append(k)

        print("Len of Prediction: " + str(len(pred_text_final)))

        dump_pickle(directory_port , pred_text_final, 'predict.d'+str(i)+'.final')
        print('-------------Prediction-------------------')
        print(pred_text[:20])
        actc_text = predicted_text(test_eng_enc_seq[:20], eng_tok)
        print(actc_text)
        print(preds[:20])
        #pred_df = pd.DataFrame({'actual' : actc_text, 'predicted' : pred_text})
        #pred_df.sample(15)
