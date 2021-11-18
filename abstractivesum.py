import numpy as np
import os
import pandas as pd
import re
from bs4 import BeautifulSoup 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from attention import AttentionLayer
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import warnings

data=pd.read_csv("reviews.csv")
contraction_mapping = {"ain't": "is not",
                   "aren't": "are not",
                   "can't": "cannot",
                   "'cause": "because",
                   "could've": "could have",
                   "couldn't": "could not",
                   "didn't": "did not",
                   "doesn't": "does not",
                   "don't": "do not",
                   "hadn't": "had not",
                   "hasn't": "has not",
                   "haven't": "have not",
                   "he'd": "he would",
                   "he'll": "he will",
                   "he's": "he is",
                   "how'd": "how did",
                   "how'd'y": "how do you",
                   "how'll": "how will",
                   "how's": "how is",
                   "I'd": "I would",
                   "I'd've": "I would have",
                   "I'll": "I will",
                   "I'll've": "I will have",
                   "I'm": "I am",
                   "I've": "I have",
                   "i'd": "i would",
                   "i'd've": "i would have",
                   "i'll": "i will",
                   "i'll've": "i will have",
                   "i'm": "i am",
                   "i've": "i have",
                   "isn't": "is not",
                   "it'd": "it would",
                   "it'd've": "it would have",
                   "it'll": "it will",
                   "it'll've": "it will have",
                   "it's": "it is",
                   "let's": "let us",
                   "ma'am": "madam",
                   "mayn't": "may not",
                   "might've": "might have",
                   "mightn't": "might not",
                   "mightn't've": "might not have",
                   "must've": "must have",
                   "mustn't": "must not",
                   "mustn't've": "must not have",
                   "needn't": "need not",
                   "needn't've": "need not have",
                   "o'clock": "of the clock",
                   "oughtn't": "ought not",
                   "oughtn't've": "ought not have",
                   "shan't": "shall not",
                   "sha'n't": "shall not",
                   "shan't've": "shall not have",
                   "she'd": "she would",
                   "she'd've": "she would have",
                   "she'll": "she will",
                   "she'll've": "she will have",
                   "she's": "she is",
                   "should've": "should have",
                   "shouldn't": "should not",
                   "shouldn't've": "should not have",
                   "so've": "so have",
                   "so's": "so as",
                   "this's": "this is",
                   "that'd": "that would",
                   "that'd've": "that would have",
                   "that's": "that is",
                   "there'd": "there would",
                   "there'd've": "there would have",
                   "there's": "there is",
                   "here's": "here is",
                   "they'd": "they would",
                   "they'd've": "they would have",
                   "they'll": "they will",
                   "they'll've": "they will have",
                   "they're": "they are",
                   "they've": "they have",
                   "to've": "to have",
                   "wasn't": "was not",
                   "we'd": "we would",
                   "we'd've": "we would have",
                   "we'll": "we will",
                   "we'll've": "we will have",
                   "we're": "we are",
                   "we've": "we have",
                   "weren't": "were not",
                   "what'll": "what will",
                   "what'll've": "what will have",
                   "what're": "what are",
                   "what's": "what is",
                   "what've": "what have",
                   "when's": "when is",
                   "when've": "when have",
                   "where'd": "where did",
                   "where's": "where is",
                   "where've": "where have",
                   "who'll": "who will",
                   "who'll've": "who will have",
                   "who's": "who is",
                   "who've": "who have",
                   "why's": "why is",
                   "why've": "why have",
                   "will've": "will have",
                   "won't": "will not",
                   "won't've": "will not have",
                   "would've": "would have",
                   "wouldn't": "would not",
                   "wouldn't've": "would not have",
                   "y'all": "you all",
                   "y'all'd": "you all would",
                   "y'all'd've": "you all would have",
                   "y'all're": "you all are",
                   "y'all've": "you all have",
                   "you'd": "you would",
                   "you'd've": "you would have",
                   "you'll": "you will",
                   "you'll've": "you will have",
                   "you're": "you are",
                   "you've": "you have"
                   }

stop_words = set(stopwords.words('english')) 
def text_cleaner(text):
    #newString = str(text)
    #newString = newString.lower()
    #newString = BeautifulSoup(newString, "lxml").text
    newString = text.lower()
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

cleaned_text = []
for t in data['News']:
    cleaned_text.append(text_cleaner(t))

def summary_cleaner(text):
    newString = re.sub('"','', text)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = newString.lower()
    tokens=newString.split()
    newString=''
    for i in tokens:
        if len(i)>1:                                 
            newString=newString+i+' '  
    return newString

#Call the above function
cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(summary_cleaner(t))

data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary
data['cleaned_summary'].replace('', np.nan, inplace=True)
data.dropna(axis=0,inplace=True)

max_len_text=80 
max_len_summary=10


x_tr,x_val,y_tr,y_val=train_test_split(data['cleaned_text'],data['cleaned_summary'],test_size=0.1,random_state=0,shuffle=True) 
 
K.clear_session() 
latent_dim = 500 

# Encoder 
encoder_inputs = Input(shape=(max_len_text,)) 
enc_emb = Embedding(1000, latent_dim,trainable=True)(encoder_inputs) 

#Preparing LSTM layer 1 
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) 
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb) 

#Preparing LSTM layer 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True) 
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 

#Preparing LSTM layer 3
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True) 
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 

# Decoder layer 
decoder_inputs = Input(shape=(None,)) 
dec_emb_layer = Embedding(2000, latent_dim,trainable=True) 
dec_emb = dec_emb_layer(decoder_inputs) 


decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) 
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

#Preparing the Attention Layer
#attn_layer = AttentionLayer(name='attention_layer') 
#attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs]) 
#decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Adding the dense layer
decoder_dense = TimeDistributed(Dense(1000, activation='softmax')) 
#decoder_outputs = decoder_dense(decoder_concat_input) 

# Prepare the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 

# Compiling the RNN model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

def decode_sequence(input_sequence):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_sequence)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!='end'):
            decoded_sentence += ' '+sampled_token

            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary-1)):
                stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence
## Making the seq2seq summary
def seq2seqsummary(input_sequence):
    newString=''
    for i in input_sequence:
      if((i!=0 and i!=target_word_index['start']) and i!=target_word_index['end']):
        newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_sequence):
    newString=''
    for i in input_sequence:
      if(i!=0):
        newString=newString+reverse_source_word_index[i]+' '
    return newString

for i in range(len(x_val)):
  print("Predicted summary:",decode_sequence(x_val[i].reshape(1,max_len_text)))
  print("\n")

