#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from string import punctuation
import json


# ## Data Preprocessing

# In[2]:


# Load data
with open("./sentiment.txt") as f:
    reviews = f.read()
    
reviews


# In[3]:


data = pd.DataFrame([review.split('\t') for review in reviews.split('\n')], columns=['Review', 'Sentiment'])
data.head()


# In[4]:


# Shuffle the data
data = data.sample(frac=1)
data.head()


# In[5]:


def split_words_reviews(data):
    text = list(data['Review'].values)
    clean_text = []
    for t in text:
        clean_text.append(t.translate(str.maketrans('', '', punctuation)).lower().rstrip())
    tokenized = [word_tokenize(x) for x in clean_text]
    all_text = []
    for tokens in tokenized:
        for t in tokens:
            all_text.append(t)
    return tokenized, set(all_text)


# In[6]:


# Tokenize
reviews, vocab = split_words_reviews(data)

print(reviews[0])
print(vocab)


# In[7]:


def create_dictionaries(words):
    word_to_int_dict = {w:i+1 for i, w in enumerate(words)}
    word_to_int_dict[''] = 0  # Ensuring the empty string is included and mapped to 0
    int_to_word_dict = {i:w for w, i in word_to_int_dict.items()}
    return word_to_int_dict, int_to_word_dict


# In[8]:


# Creating vocabulary
word_to_int_dict, int_to_word_dict = create_dictionaries(vocab)

print(int_to_word_dict)
print(word_to_int_dict)


# In[9]:


with open('word_to_int_dict.json', 'w') as fp:
    json.dump(word_to_int_dict, fp)


# In[10]:


print(np.max([len(x) for x in reviews]))
print(np.mean([len(x) for x in reviews]))


# In[11]:


def pad_text(tokenized_reviews, seq_length):
    reviews = []
    for review in tokenized_reviews:
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append(['']*(seq_length-len(review)) + review)
    return np.array(reviews)


# In[12]:


# Pad and encode reviews
seq_length = 50
padded_sentences = pad_text(reviews, seq_length)
encoded_sentences = np.array([[word_to_int_dict[word] for word in review] for review in padded_sentences])
encoded_sentences[0]


# In[13]:


# Prepare labels
labels = np.array([int(x) for x in data['Sentiment'].values])


# ## Model Building

# In[14]:


class SentimentBiLSTM(Model):
    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p=0.8):
        super(SentimentBiLSTM, self).__init__()
        self.embedding = layers.Embedding(input_dim=n_vocab, output_dim=n_embed)
        self.lstm = layers.Bidirectional(
            layers.LSTM(n_hidden, return_sequences=True, dropout=drop_p)
        )
        self.dropout = layers.Dropout(drop_p)
        self.fc = layers.Dense(n_output, activation='sigmoid')

    def call(self, x):
        x = self.embedding(x)
        lstm_out = self.lstm(x)  # Only one value is returned
        x = self.dropout(lstm_out)
        x = self.fc(x)
        return x[:, -1]

n_vocab = len(word_to_int_dict)
n_embed = 50
n_hidden = 100
n_output = 1
n_layers = 2

# Instantiate the model
model = SentimentBiLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])


# ## Model Training

# In[15]:


train_ratio = 0.8
valid_ratio = (1 - train_ratio) / 2

total = len(encoded_sentences)
train_cutoff = int(total * train_ratio)
valid_cutoff = int(total * (1 - valid_ratio))

train_x, train_y = encoded_sentences[:train_cutoff], labels[:train_cutoff]
valid_x, valid_y = encoded_sentences[train_cutoff:valid_cutoff], labels[train_cutoff:valid_cutoff]
test_x, test_y = encoded_sentences[valid_cutoff:], labels[valid_cutoff:]

batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)


# In[16]:


model.fit(train_dataset, validation_data=valid_dataset, epochs=15)


# In[17]:


model.summary()


# ## Model Evaluation and Prediction

# In[18]:


def preprocess_review(review, word_to_int_dict, seq_length=50):
    review = review.translate(str.maketrans('', '', punctuation)).lower().rstrip()
    tokenized = word_tokenize(review)
    if len(tokenized) >= seq_length:
        review = tokenized[:seq_length]
    else:
        review = ['']*(seq_length-len(tokenized)) + tokenized

    final = []
    for token in review:
        try:
            final.append(word_to_int_dict[token])
        except:
            final.append(word_to_int_dict[''])
    return np.array(final)


# In[19]:


test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


# In[20]:


def predict_review(review):
    review_encoded = preprocess_review(review, word_to_int_dict)
    review_encoded = np.expand_dims(review_encoded, axis=0)
    prediction = model.predict(review_encoded)
    msg = "This is a positive review." if prediction >= 0.5 else "This is a negative review."
    print(msg)
    print(f'Prediction = {prediction[0][0]}')


# In[21]:


predict_review("Loved the casting of Jimmy Buffet as the science teacher.")
predict_review("There was absolutely no warmth or charm to these scenes or characters.")


# In[24]:


predict_review("I think the movie was up to the mark.")

