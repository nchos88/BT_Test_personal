import csv
import io
import sys
import numpy as np
import pandas as pd
import re

path1 ='refined_category_dataset.csv'

def Loader(path):
    pddata = pd.read_csv(path  ,  header=None, sep = "\"\," , encoding = 'utf-8')
    dicts = pddata.transpose().to_dict("list").values()
    alldatalist = []
    
   
    for line in dicts:
        for i in range(len(line)):
            line[i] = line[i].replace('{','').replace('}','').replace('"','')
            temp = line[i]
         
    
        res = [ item.split(',') for item in line ]
    
        tempdic = {}
    
        for item , i in zip(res , range(len(res))):
            splited = None
            if i == 1 :
                splited = item[0].split('img:')
                tempdic['img'] = splited[1]
            else :
                splited = item[0].split(':')
                tempdic[splited[0].replace(' ' , '')] = splited[1]
        
    
        alldatalist.append(tempdic)
    return alldatalist

def SelectCol(data,col):
    output = [ item[col] for item in data ] 
    return output

def Trim(wordlist):
    output = re.sub('[-+=.#/?:$}{}()]' , '' , wordlist)
    output = output.replace('[' , '' )
    output = output.replace(']' , '' )
    output = output.replace('_' , ' ' )
    for i in range(10):
        output = output.replace(str(i) , ' ' )
    return output



