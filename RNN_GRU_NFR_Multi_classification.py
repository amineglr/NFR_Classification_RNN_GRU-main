#!/usr/bin/env python
# coding: utf-8

# In[2]:


from psutil import virtual_memory
from tensorflow.python.client import device_lib
from numpy import where
from keras import backend as K
import plotly.graph_objs as go
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, classification_report, confusion_matrix, multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import KFold
import sklearn.datasets
from sklearn.feature_extraction import _stop_words
from nltk.stem.lancaster import LancasterStemmer
from nltk import punkt
import nltk
from scipy import sparse
from keras.layers import Dropout
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GRU
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import datetime
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import tensorflow as tf
from collections import Counter
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import re
from os.path import exists
import os
import sys
import gensim
print(sys.path)
# from keras.callbacks import EarlyStopping
# use natural language toolkit
nltk.download('punkt')
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


# In[3]:


# connect to local google drive if using google colab by un-commenting the below commands
# from google.colab import drive
# drive.mount('/content/drive')


# In[4]:


ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
if ram_gb < 20:
    print('You are using a normal-RAM runtime!')
else:
    print('You are using a high-RAM runtime!')


# In[5]:


# check if GPU is used
print(device_lib.list_local_devices())


# In[ ]:


# import the word2vec model with pre-trained vectors trained on part of Google News dataset (about 100 billion words).
# The model contains 300-dimensional vectors for 3 million words and phrases. Link here to the official web page for this project -- https://code.google.com/archive/p/word2vec/
# Per the official website link above, one could download the model as a gz file from the archives here -- https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit (or)
vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    'C:/Users/AyhanÇavdar/Desktop/SWE522-GroupH-A3/NFR_Classification_RNN_GRU-main/GoogleNews-vectors-negative300.bin', binary=True)


# In[6]:


ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
if ram_gb < 20:
    print('You are using a normal-RAM runtime!')
else:
    print('You are using a high-RAM runtime!')


# In[ ]:


print(vec_model['the'])


# In[ ]:


vocab_size = len(vec_model.index_to_key)
print(vocab_size)


# In[7]:


# import the dataset from the Dataset folder
df = pd.read_csv(
    '"C:/Users/AyhanÇavdar/Desktop/SWE522-GroupH-A3/NFR_Classification_RNN_GRU-main/NFR_CSV.csv"')
df.info()


# In[8]:


df.class_name.value_counts()


# In[9]:


# function to print out a specific observation by passing the index of that in the dataset
def print_plot(index):
    example = df[df.index == index][['sentence', 'class_name']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Req_Class:', example[1])


print_plot(10)


# In[10]:


print_plot(100)


# In[ ]:


df = df.reset_index(drop=True)
# Regex options to find the special characters
REPLACE_BY_SPACE = re.compile('[/(){}\[\]\|@,;]')
# Regex options to find the bad symbols
BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')
# Import stopwords from the nltk.corpus and apply to the data
STOPWORDS = set(stopwords.words('english'))
# Custom words found by manually looking at the dataset observations
ignore_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
    '?', '%', '/', '(', ')', '[', ']', '-', ':', ';', 'system', 'product',
    'application', 'should', 'would', 'shall', 'go', 'System', 'system.',
    'System.', 'â€', "'", ]

# function to perform data cleansing steps to convert to lower case, remove stop words, and to ignore custom words from the list manually gleaned as shown above


def clean_text(text):
    text = text.lower()  # lowercase text
    # replace REPLACE_BY_SPACE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE with space.
    text = REPLACE_BY_SPACE.sub(' ', text)
    # remove symbols which are in BAD_SYMBOLS from text. substitute the matched string in BAD_SYMBOLS with nothing.
    text = BAD_SYMBOLS.sub('', text)
    text = text.replace('x', '')
    # remove stopwords from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    # remove words unique to requirements dataset
    text = ' '.join(word for word in text.split() if word not in ignore_words)
    return text


# apply the function on the df from previous step
df['sentence'] = df['sentence'].apply(clean_text)


# In[ ]:


# Setting the maximum number of words in the overall dataset.
word_vectors = vec_model
MAX_NB_WORDS = 50000
# Setting the ceiling for the maximum number of words in each requirement.
MAX_SEQUENCE_LENGTH = 250
# Setting the number of dimensions to be used, since this word2vec model uses 300, this is fixed.
EMBEDDING_DIM = 300
# Invoke Tokenizer function by creating an instance of it
tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# Fitting the instance on the data
tokenizer.fit_on_texts(df['sentence'].values)
# Saving the output of the tokenizer into different variables
word_index = tokenizer.word_index
word_counts = tokenizer.word_counts
word_docs = tokenizer.word_docs
# Prints the unique number of tokens which are basically unique words after the cleaning process and tokenization process
print('Found %s unique tokens.' % len(word_index))
# This print each unique word and the number of occurrences of each word
print(word_docs)
print(word_index)
# prints the unique words parsed from the nth requirement
print_plot(1)


# In[ ]:


# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)


# In[ ]:


# Transforms each text in df['sentence'] to a sequence of integers
X = tokenizer.texts_to_sequences(df['sentence'].values)
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


# In[ ]:


# Find the word from the data in the word2vec model and pass the vector back
def getVector(str):
    if str in vec_model:
        return vec_model[str]
    else:
        return None


# In[ ]:


# Below are steps required to identify each unique token (word parsed and cleaned) from the tokenizer to obtain the word2vec vector matrix to be used in the subsequent model development
# set the matrix size to the number of unique tokens + 1 to avoid out of bounds error
wsize = len(tokenizer.word_index)+1
# Initialize the weighting matrix to be used in the model below
wv_matrix = np.zeros((wsize, 300))
# for every unique word in the tokenizer get the vector from word2vec model
for word, i in tokenizer.word_index.items():
    embedding_vector = getVector(word)
    if embedding_vector is not None:
        wv_matrix[i] = embedding_vector


# In[ ]:


print(len(wv_matrix))
# print the unique weighted matrix value for the nth word as received by running it through the word2vec model
print(wv_matrix[7])


# In[ ]:


# Y = pd.get_dummies(df['class_name'])
Y = pd.get_dummies(df['class_name']).values
print('Shape of label tensor:', Y.shape)
y, y_uniques = pd.factorize(df['class_name'], sort=True)
print(y)
print(Y)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, random_state=34, stratify=Y)
X_train_n, X_test_n, Y_train_n, Y_test_n = train_test_split(
    X, y, test_size=0.10, random_state=34, stratify=Y)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(X_train_n.shape, Y_train_n.shape)
print(X_test_n.shape, Y_test_n.shape)
print(Y_train, Y_test)
print(Y_train_n, Y_test_n)
# print(y_tr_unique)
original_text_t = []
original_text = list(map(sequence_to_text, X_test))
for i in range(0, len(original_text)):
    temp = []
    for j in range(0, len(original_text[i])):
        if(original_text[i][j] != None):
            temp.append(original_text[i][j])
    original_text_t.append(temp)
# print(my_texts_t)
# print(len(my_texts_t))


# In[ ]:


def test_class(i):
    switcher = {
        0: 'Main',
        1: 'Oper',
        2: 'Perf',
        3: 'Secu',
        4: 'Usab'
    }
    return switcher.get(i, "Invalid class")


# In[ ]:

def create_model_GRU():
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM,
              weights=[wv_matrix], trainable=False, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
                  'acc', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


""" def create_model_LSTM():
    #The TensorFlow Keras API makes easy to build models and experiment while Keras handles the complexity of connecting everything together. 
    #The tf.keras.Sequential model is a linear stack of layers. 
    #In this case, one LSTM layer with 100 nodes each, and an output layer with 5 nodes representing our label predictions. 
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[wv_matrix], trainable=False, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])    
    print(model.summary())
    return model
 """
# In[ ]:


def make_con_mat(cf_matrix, class_name):
    plt.title("Confusion Matrix Heat Map for class: %s" % class_name)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


# In[ ]:


print(len(wv_matrix))


# In[ ]:


# fix random seed for reproducibility
seed = 123
batch_size = 30
# define 10-fold cross validation test
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscorespre = []
cvscoresrec = []
cvscoresauc = []
cvscoresf1 = []
for train_index, val_index in kfold.split(X_train, Y_train_n):
    print("TRAIN:", train_index, "VAL:", val_index)
#     X_train_k, X_val, Y_train_k, Y_val = train_test_split(X, Y_train, test_size = 0.10, random_state = 34)
    X_train_k, X_val = X_train[train_index], X_train[val_index]
    Y_train_k, Y_val = Y_train[train_index], Y_train[val_index]
    print(X_train_k, X_val, Y_train_k, Y_val)
    model = create_model_GRU()
    history = model.fit(X_train_k, Y_train_k, epochs=20, batch_size=batch_size, validation_split=0.1, callbacks=[
                        EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
#     history=model.fit(X_train_k, Y_train_k,epochs=10,batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
#     history=model.fit(X_train, Y_train,epochs=50,batch_size=batch_size,validation_split=0.1)
    scores = model.evaluate(X_val, Y_val)
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
    print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))
    cvscoresauc.append(scores[2] * 100)
    cvscorespre.append(scores[3] * 100)
    cvscoresrec.append(scores[4] * 100)
    cvscoresf1.append(
        2*(scores[3]*scores[4])/(scores[3]+scores[4]+tf.keras.backend.epsilon()) * 100)
    Y_pred = model.predict(X_val)
    Y_pred_cm = (Y_pred.argmax(1)[:, None] ==
                 np.arange(Y_pred.shape[1])).astype(int)
    con_mat = multilabel_confusion_matrix(
        Y_val, Y_pred_cm, labels=[0, 1, 2, 3, 4])
    # plt.figure()
    # print("****************************Confustion Matrix Plots Start here for a new fold**************")
    # class_name = "Maintainability"
    # make_con_mat(con_mat[0],class_name)
    # plt.figure()
    # class_name = "Operability"
    # make_con_mat(con_mat[1], class_name)
    # plt.figure()
    # class_name = "Performance"
    # make_con_mat(con_mat[2],class_name)
    # plt.figure()
    # class_name = "Security"
    # make_con_mat(con_mat[3], class_name)
    # plt.figure()
    # class_name = "Usability"
    # make_con_mat(con_mat[4],class_name)


# In[ ]:


# Calculating Mean Average of model scores
print("Mean & std for Area under Curve: %.2f%% (+/- %.2f%%)" %
      (np.mean(cvscoresauc), np.std(cvscoresauc)))
print("Mean & std for Precision: %.2f%% (+/- %.2f%%)" %
      (np.mean(cvscorespre), np.std(cvscorespre)))
print("Mean & std for Recall: %.2f%% (+/- %.2f%%)" %
      (np.mean(cvscoresrec), np.std(cvscoresrec)))
print(("Mean & std for F1 score: %.2f%% (+/- %.2f%%)" %
      (np.mean(cvscoresf1), np.std(cvscoresf1))))


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("Maximum Training Accuracy: %s" % np.max(history.history['acc']))
print("Maximum Validation Accuracy: %s" % np.max(history.history['val_acc']))
print("Minimum Training Loss: %s" % np.min(history.history['loss']))
print("Minimum Validation Loss: %s" % np.min(history.history['val_loss']))

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}\n  AUC: {:0.3f}\n  Precision: {:0.3f}\n Recall: {:0.3f}\n'.format(
    accr[0], accr[1], accr[2], accr[3], accr[4]))
Y_pred_test = model.predict(X_test)
print(Y_pred_test)
Y_pred_test_cm = (Y_pred_test.argmax(1)[:, None] == np.arange(
    Y_pred_test.shape[1])).astype(int)
con_mat_test = multilabel_confusion_matrix(
    Y_test, Y_pred_test_cm, labels=[0, 1, 2, 3, 4])
plt.figure()
print("****************************Confustion Matrix Plots Start here for Test Data**************")
class_name = "Maintainability"
make_con_mat(con_mat_test[0], class_name)
plt.figure()
class_name = "Operability"
make_con_mat(con_mat_test[1], class_name)
plt.figure()
class_name = "Performance"
make_con_mat(con_mat_test[2], class_name)
plt.figure()
class_name = "Security"
make_con_mat(con_mat_test[3], class_name)
plt.figure()
class_name = "Usability"
make_con_mat(con_mat_test[4], class_name)


# In[ ]:


Y_pred_main = [i[0] for i in Y_pred_test]
Y_pred_oper = [i[1] for i in Y_pred_test]
Y_pred_perf = [i[2] for i in Y_pred_test]
Y_pred_secu = [i[3] for i in Y_pred_test]
Y_pred_usab = [i[4] for i in Y_pred_test]
Y_test_main = [i[0] for i in Y_test]
Y_test_oper = [i[1] for i in Y_test]
Y_test_perf = [i[2] for i in Y_test]
Y_test_secu = [i[3] for i in Y_test]
Y_test_usab = [i[4] for i in Y_test]
Y_test_class = []
for i in range(0, len(Y_test)):
    for j in range(0, len(Y_test[i])):
        if (Y_test[i][j] == 1):
            Y_test_class.append(test_class(j))
print(Y_test_class)
Y_pred_class = []
for i in range(0, len(Y_pred_test_cm)):
    for j in range(0, len(Y_pred_test_cm[i])):
        if (Y_pred_test_cm[i][j] == 1):
            Y_pred_class.append(test_class(j))
print(Y_pred_class)
print(len(Y_test_class))
print(len(Y_pred_class))
print(len(original_text_t))


# In[ ]:


# combine y_test, x_test, y_pred_test_cm
combined_output = list(zip(original_text_t, Y_test_main, Y_test_oper, Y_test_perf, Y_test_secu, Y_test_usab, Y_test_class, Y_pred_main, Y_pred_oper, Y_pred_perf,
                           Y_pred_secu, Y_pred_usab, Y_pred_class))
combined_output
dfo = pd.DataFrame(combined_output, columns=['X_test', 'Y_test_main', 'Y_test_oper', 'Y_test_perf', 'Y_test_secu', 'Y_test_usab',
                                             'Y_test_class', 'Y_pred_main', 'Y_pred_oper', 'Y_pred_perf', 'Y_pred_secu', 'Y_pred_usab',
                                             'Y_pred_class'])
dfo.to_csv('out.csv', index=False)


# # New Section

# In[ ]:


new_requirement = [
    'Passwords should be held to a standard and be required to change at intervals.']
seq = tokenizer.texts_to_sequences(new_requirement)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['Maintainability', 'Operability',
          'Performance', 'Security', 'Usability']
print(pred, labels[np.argmax(pred)])


# ***************Some try out code below: Commented out in case if it would be needed later

# In[ ]:


# #mapping target variable's classes to unique labels and then to one hot labels
# labels = sorted(list(set(df['class_name'].tolist())))
# one_hot = np.zeros((len(labels), len(labels)), int)
# np.fill_diagonal(one_hot, 1)
# label_dict = dict(zip(labels, one_hot))
# num_labels = []
# for z in range(len(labels)):
#     num_labels.append(z)

# num_label_dict = dict(zip(labels, num_labels))

# x_raw = df['sentence'].apply(lambda x: clean_text(x)).tolist()
# y_raw = df['class_name'].apply(lambda y: label_dict[y]).tolist()
# y_raw_num = df['class_name'].apply(lambda y: num_label_dict[y]).tolist() #current class stored as 1 - as before
# y = np.array(y_raw)
# y_num = np.array(y_raw_num)
# print(y,y_num)


# In[ ]:


# print(X_test)
# for value in range(0,len(X_test)):
#     X_test_1[value] = X_test[value] != 0
#     print(X_test_0[value])
# # print(Y_test_text)
# # X_test_0 = X_test[X_test != 0]
# # print(X_test_0)
# # X_test_text = list(map(sequence_to_text, X_test_0))
# # print(X_test_text)


# In[ ]:


# Y_test_text =  list(map(sequence_to_text, Y_test))
# print(Y_test_text)
# Y_test_cm_text = list(map(sequence_to_text, Y_pred_test_cm))
# print(Y_test_cm_text)
# X_test_text = list(map(sequence_to_text, X_test))
# print(X_test_text)


# In[ ]:


# # Y_test_no_0 = list(filter(lambda num: num != 0, Y_test))
# Y_test_text = Y_test
# for value in range(0,len(Y_test_text)):
#     if Y_test_text[value] <= 0:
#         del Y_test_text[value]
#         print(Y_test_text)
# print(Y_test_text)
# # Y_test_text =  list(map(sequence_to_text, Y_test_no_0))
# # print(Y_test_text)
# # Y_test_cm_text = list(map(sequence_to_text, Y_pred_test_cm))
# # print(Y_test_cm_text)
# # X_test_text = list(map(sequence_to_text, X_test))
# # print(X_test_text)


# In[ ]:


# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
# estimator = KerasClassifier(build_fn=create_model, epochs=3, batch_size=60, verbose=0)
# kfold = KFold(n_splits=2, shuffle=True)
# results = cross_val_score(estimator, X_train, Y_train, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# print(results)


# In[ ]:


# # encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(df['class_name'])
# encoded_Y = encoder.transform(df['class_name'])
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y = np_utils.to_categorical(encoded_Y)
# print(dummy_y)


# In[ ]:


# y_pred=model.predict_classes(X_test)
# y_pred=pd.get_dummies(y_pred)
# print(X_test, y_pred)
# print(Y_test)
# multilabel_confusion_matrix(Y_test, y_pred, *, labels=['Performance', 'Usability', 'Security', 'Operability', 'Maintainability'])


# In[ ]:


# plt.title('Loss')
# plt.plot(cvscoreslos, label='train')
# plt.plot((accr[0]*100), label='test')
# plt.legend()
# plt.show();


# In[ ]:


# plt.title('Accuracy')
# plt.plot(cvscoresacc, label='train')
# plt.plot(accr[1]*100, label='test')
# plt.legend()
# plt.show();


# In[ ]:


# epochs = 10
# batch_size = 60

# history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# In[ ]:
