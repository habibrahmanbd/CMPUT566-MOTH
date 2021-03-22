
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

from keras import backend as K
from keras import __version__
print('Using Keras version:', __version__, 'backend:', K.backend())

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
# prepare english tokenizer
eng_tokenizer = load_clean_sentences('en1.pickle') #this is /tokenized/English/1.pickle
#eng_vocab_size = len(eng_tokenizer.word_index) + 1 #this is a piece of info I am missing
eng_vocab_size = 4000
#eng_length = max_length(dataset[:, 0]) # no. based on proj mid report
eng_length = 15
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare portugese tokenizer
por_tokenizer = load_clean_sentences('pt1.pickle') #this is /tokenized/Portuguese/1.pickle
#por_vocab_size = len(por_tokenizer.word_index) + 1 #this is a piece of info I am missing
por_vocab_size = 4000
#por_length = max_length(dataset[:, 1]) # no. based on proj mid report
por_length = 15
print('Portugese Vocabulary Size: %d' % por_vocab_size)
print('Portugese Max Length: %d' % (por_length))

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, hidden_size):
  use_dropout = True
  model = Sequential()
  model.add(Embedding(src_vocab, hidden_size, input_length = src_timesteps))
  model.add(LSTM(hidden_size))
  model.add(RepeatVector(tar_timesteps))
  model.add(LSTM(hidden_size, return_sequences=True))
  if use_dropout:
    model.add(Dropout(0.5))
  model.add(TimeDistributed(Dense(tar_vocab, activation = 'softmax')))
  
  return model

# define model
model = define_model(eng_vocab_size, por_vocab_size, eng_length, por_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
