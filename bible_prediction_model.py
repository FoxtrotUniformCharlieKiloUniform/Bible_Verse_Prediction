import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import csv
import pandas as pd
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import pad_sequences
import pickle
from keras import callbacks
import keras.backend as K
from keras import Model
from keras.layers import Layer
import keras.layers as layers
import keras.backend as K


batch_size=1
number_of_verses = 20

#Goal 1: assign my input text to a verse

#import data 
directory="archive/Bible/asv_bible.csv"

bible_csv_before_processing = pd.read_csv(
    directory,
    names=["verse id","book name", "book number", "chapter", "verse","text"]
)

#FOR TESTING ONLY
bible_csv_before_processing = bible_csv_before_processing.head(number_of_verses+1)

print(bible_csv_before_processing)

num_rows,num_cols = bible_csv_before_processing.shape

print(f"Data has {num_rows} rows and {num_cols} cols")

#shoutout GPT, I didn't write any of the following lines

# List of words to remove
words_to_remove = ['an', 'like', 'is', 'the', 'in','and']
words_to_remove = ['xyzabc']
# Create a regex pattern that matches any of these words
pattern = r'\b(?:' + '|'.join(words_to_remove) + r')\b'
# Replace the words with an empty string
bible_csv_before_processing['text'] = bible_csv_before_processing['text'].str.replace(pattern, '', regex=True)
# Remove extra spaces that may have resulted from the replacement
bible_csv_before_processing['text'] = bible_csv_before_processing['text'].str.replace(r'\s+', ' ', regex=True).str.strip()


bible_csv = bible_csv_before_processing.iloc[0:]     #upgrade! we've now processed the data
#bible_csv['text']=str(bible_csv_before_processing['text'])

#alright, back to me
#i'm going to use keras.layers.Embedding in the NN architecture in the second half (myself) but right now I'm following a guide

#transform into tuple of verse id, text. For output, will construct based on that verse id
tokenizer = keras.preprocessing.text.Tokenizer(oov_token = '<oov>')
tokenizer.fit_on_texts(bible_csv['text'])
sequences = tokenizer.texts_to_sequences(bible_csv['text'])

word_index = tokenizer.word_index
print("tokenizer word index is ", word_index)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('sequences.pickle', 'wb') as handle:
    pickle.dump(sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)



total_words = len(tokenizer.word_index) + 1

output_labels = bible_csv['verse id']


# Padding! Makes sure input is always the same length
input_sequences = sequences

max_sequence_len = max([len(x) for x in input_sequences])
print(f"Max sequence length is {max_sequence_len}")
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences = np.delete(input_sequences,(0),axis=0)

output_labels = np.delete(output_labels,(0),axis=0)
print("input sequences size is ",tf.size(input_sequences))


#xs = np.delete(np.array(input_sequences),(0),axis=0)
xs = np.array(input_sequences)
labels = output_labels
ys = tf.keras.utils.to_categorical(labels)

#tf.print(xs)
print("\n")
#tf.print(labels)
print("\n")
#tf.print(ys)
print(len(output_labels))


#This architecture ran, but with extraordinarily low accuracy. 

model = keras.Sequential()
model.add(layers.Masking(mask_value=0))     #in tandem with no LSTM. delete if going back to old one
#model.add(keras.Input(shape=(max_sequence_len,)))
model.add(layers.Embedding(total_words, 100, input_length=max_sequence_len))
model.add(layers.Flatten())  #in tandem with no LSTM. delete if going back to old one
#model.add(layers.Bidirectional(layers.LSTM(256)))
model.add(layers.Dense(len(output_labels)+1, activation='softmax'))



class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
    def build(self,input_shape):
        self.w=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention,self).build(input_shape)
        
        
    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.w)+self.b), axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])
    def get_config(self):
        return super(attention,self).get_config() 
    
#LSTM with attention - broken
'''
model = keras.Sequential()
model.add(keras.Input(shape=(max_sequence_len,)))
model.add(layers.Embedding(total_words, 100, input_length=max_sequence_len))
model.add(layers.Bidirectional(layers.LSTM(len(output_labels)*25,activation='relu',return_sequences = True)))
model.add(attention())
model.add(layers.Dense(len(output_labels)+1, activation='softmax'))
'''

#LSTM without attention 
'''
model = keras.Sequential()
model.add(keras.Input(shape=(max_sequence_len,)))
model.add(layers.Embedding(total_words, 100, input_length=max_sequence_len))
model.add(layers.Bidirectional(layers.LSTM(len(output_labels)*25,activation='relu',return_sequences = True)))
model.add(layers.Flatten())
model.add(layers.Dense(len(output_labels)+1, activation='softmax'))
'''

#custom callback
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        if logs.get("accuracy")>0.90:
            print("Accuracy over 90%, quitting training")
            self.model.stop_training=True
        if np.isnan(logs.get("loss")):
            print("Loss is NaN, quitting training (gradient vanished)")
            self.model.stop_training=True


adam = keras.optimizers.Adam(lr=0.00001)
model.compile(
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
              optimizer=adam, 
              metrics=['accuracy'],
              )
history = model.fit(xs, ys, epochs=1000, verbose=1,callbacks = [CustomCallback()])
print(model.summary())                                      
print(model)                                                            
model.save_weights('biblepredictionmodelweights/')                                      
model.save('biblepredictionmodel.keras')                                        


print(f"max_sequence_len = {max_sequence_len}, total words is {total_words} and len(outpUtLabels+1) is {len(output_labels)+1}")

