import csv
import io
import sys
import numpy as np
import pandas as pd
import chardet
import DataLoader as dl
import nltk
from gensim.models import word2vec

from konlpy.tag import Twitter , Kkma
from konlpy.utils import pprint

path1 = 'refined_category_dataset.csv' 
#pddata = pd.read_csv('refined_category_dataset.csv'  ,  header=None, sep = "\"\," , encoding = 'utf-8')
#dicts = pddata.transpose().to_dict("list").values()
alldatalist = dl.Loader(path1)

def term_exists(doc,base):
    return {'exists({})'.format(word): (word in set(doc)) for word in base}



#val = alldatalist[0]['name']
#val = dl.Trim(val)

# extract only word
t = Twitter()
#kkma = Kkma()
#tokens_ko = kkma.nouns(val)
wordlist = [ dl.Trim( item['name'] ) for item in alldatalist ]

tokens_ko = []
templist = []
for i in range(len(wordlist)):
    temp = t.pos(wordlist[i], norm=True, stem=True)
    categ = alldatalist[i]['cate']
    tokens_ko.append((temp,categ))


temp = tokens_ko[0][0][0]
sentenselist = [ [ one[0] for one in item[0]] for item in tokens_ko ]

train = sentenselist[:6000]
test = sentenselist[6000:]

import gensim
from gensim.models import Word2Vec
min_count = 2
size = 50
window = 4

model = gensim.models.Word2Vec()
model.build_vocab(sentenselist)
model.train(train)



vocab = list(model.vocab.keys())
vocab[:10]

tokens = [t for d in tokens_ko for t in d[0]]
text = nltk.Text(tokens, name='NMSC')
temp = text.vocab().most_common(10)
#extract categorry 

catelist = [  item['cate']  for item in alldatalist ]
cateset = set(catelist)
catelist = list(cateset)

catelen = len(catelist)

keycateset = {}

icate = 0
for cate in catelist:
    keycateset[cate] = icate
    icate += 1

#tokens = [t for d in wordlist for t in d]
#print(len(tokens))
#text = nltk.Text(tokens_ko, name='NMSC')
#temp = text.vocab().most_common(10)
########################################
from gensim.models import Word2Vec
min_count = 2
size = 50
window = 4
 


#######################################

selected_words = [f[0] for f in text.vocab().most_common(2000)] 
train_docs = tokens_ko[:4000] 
test_docs = tokens_ko[4000:] 
train_xy = [(term_exists(d,selected_words), c) for d, c in train_docs]
test_xy = [(term_exists(d,selected_words), c) for d, c in test_docs]
classifier = nltk.NaiveBayesClassifier.train(train_xy) 
print(nltk.classify.accuracy(classifier, test_xy))

print()


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

list_classes = catelist
y = catelist
list_sentences_train = pd.DataFrame( wordlist)

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

#commented it due to long output
#for occurence of words
tokenizer.word_counts
#for index of words
tokenizer.word_index

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
plt.hist(totalNumWords,bins = np.arange(0,410,10))
plt.show()






