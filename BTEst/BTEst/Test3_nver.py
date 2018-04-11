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




tokens = [t for d in tokens_ko for t in d[0]]
text = nltk.Text(tokens, name='NMSC')
temp = text.vocab().most_common(10)
#extract categorry 

#load
from konlpy.corpus import kobill
docs_ko = [kobill.open(i).read() for i in kobill.fileids()]


#Tokenize%

from konlpy.tag import Twitter
pos = lambda d: ['/'.join(p) for p in t.pos(d)]
texts_ko = [pos(doc) for doc in docs_ko]

from gensim.models import word2vec

wv_model_ko = word2vec.Word2Vec(texts_ko)
wv_model_ko.init_sims(replace=True)
wv_model_ko.save('ko_word2vec_e.model')
temp = wv_model_ko.most_similar(pos(''))
print(temp)



catelist = [  item['cate']  for item in alldatalist ]
cateset = set(catelist)
catelist = list(cateset)

catelen = len(catelist)

keycateset = {}

icate = 0
for cate in catelist:
    keycateset[cate] = icate
    icate += 1





