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



max_sequence_len = 33
total_words = 105
lenOutputLabels = 21

'''     #WITH Attention
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
    

model = keras.Sequential()
model.add(keras.Input(shape=(max_sequence_len,)))
model.add(layers.Embedding(total_words, 100, input_length=max_sequence_len))
model.add(layers.Bidirectional(layers.LSTM((lenOutputLabels-1)*25,activation='relu',return_sequences = True)))
model.add(layers.Dense(lenOutputLabels, activation='softmax'))
model.load_weights('biblepredictionmodelweights/')
'''

model = tf.keras.models.load_model('biblepredictionmodel.keras')        #without attention

#get user input
input = input("Please enter the phrase to chzhceck ")

# List of words to remove
words_to_remove = ['an', 'like', 'is', 'the', 'in','and']
words_to_remove = ['zxcoin']
# Create a regex pattern that matches any of these words
pattern = r'\b(?:' + '|'.join(words_to_remove) + r')\b'
# Replace the words with an empty string
input = input.replace(pattern, '')
# Remove extrspaces that may have resulted from the replacement
input = input.replace(r'\s+', ' ').strip()

input = pd.DataFrame([[input]])
input = input.iloc[0:]
print(input)

tokenizer = keras.preprocessing.text.Tokenizer(oov_token = '<oov>')
tokenizer.fit_on_texts(input[0])
#total_words = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(input[0])

word_index = tokenizer.word_index
print(word_index)

#padding
input_sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

print(input_sequences)
verse_probabilities = model.predict(input_sequences)




#sum every column
#take max of that sum
sumCols = np.sum(verse_probabilities,axis=0)
print(sumCols)
verse_id = np.argmax(sumCols)
print(verse_id)
confidence = np.max(sumCols)*100
print(f"Predicted verse ID: {verse_id}, Confidence: {confidence:.2f}%")

#get chapter and verse number from predicted verse # in csv
directory="archive/Bible/asv_bible.csv"

bible_df = pd.read_csv(
    directory,
    names=["verse id","book name", "book number", "chapter", "verse","text"]
)

predicted_verse = bible_df.iloc[verse_id]
print(f"Book: {predicted_verse['book name']}, Chapter: {predicted_verse['chapter']}, Verse: {predicted_verse['verse']}")
print(f"Text: {predicted_verse['text']}")



def saveVerseProbabilities(arewe):
    if arewe == True:
        dfVP = pd.DataFrame(verse_probabilities)
        dfVP.to_csv("runs/most_recent_run_output9.csv")
    else:
        "didn't save verse probabilities shithead"

saveVerseProbabilities(True)

