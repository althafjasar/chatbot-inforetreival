# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 19:14:23 2020

@author: Jasar Althaf
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pickle
import re
from nltk.corpus import stopwords
# pd.set_option("display.max_columns",None)

def load_feature_vector():
    with open("C:\\Users\\Jasar Althaf\\Desktop\\work\\bot\\_tfidf_features.pkl",'rb') as fv:
        feature_vector = pickle.load(fv)
    return feature_vector

def load_model():
    with open('C:\\Users\\Jasar Althaf\\Desktop\\work\\bot\\_rf_model.pkl', 'rb') as m:
        mod = pickle.load(m)
    return mod


def get_category(txt):
    X = load_feature_vector()
    model = load_model()
    inp = X.transform([txt])
    o_score = model.predict(inp)
    return o_score


import re
import gensim 
from gensim.parsing.preprocessing import remove_stopwords

def clean_sentence(sentence, stopwords=False):
    
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    #sentence = re.sub(r'\s{2,}', ' ', sentence)
    
    if stopwords:
         sentence = remove_stopwords(sentence)
    
    
    return sentence
                    
def get_cleaned_sentences(df,stopwords=False):    
    sents=df[["question"]];
    cleaned_sentences=[]

    for index,row in df.iterrows():
        #print(index,row)
        cleaned=clean_sentence(row["question"],stopwords);
        cleaned_sentences.append(cleaned);
    return cleaned_sentences;

from gensim.models import Word2Vec 
import gensim.downloader as api
def load_model():    
    v2w_model=None;
    try:
        v2w_model = gensim.models.KeyedVectors.load("./w2vecmodel.mod")
    except:            
        v2w_model = api.load('word2vec-google-news-300')
        v2w_model.save("./w2vecmodel.mod")

    w2vec_embedding_size=len(v2w_model['computer'])
    return v2w_model



def getWordVec(word,model):
        samp=model['computer'];
        vec=[0]*len(samp);
        try:
                vec=model[word];
        except:
                vec=[0]*len(samp);
        return (vec)


def getPhraseEmbedding(phrase,embeddingmodel):
                       
        samp=getWordVec('computer', embeddingmodel);
        vec=np.array([0]*len(samp));
        den=0;
        for word in phrase.split():
            #print(word)
            den=den+1;
            vec=vec+np.array(getWordVec(word,embeddingmodel));
        #vec=vec/den;
        #return (vec.tolist());
        return vec.reshape(1, -1)

import sklearn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity;
def retrieveAndPrintFAQAnswer(question_text,question_embedding,sentence_embeddings,FAQdf,sentences):
    max_sim=-1;
    index_sim=-1;
    sec_max_sim=-1;
    sec_index_sim=-1;
    for index,faq_embedding in enumerate(sentence_embeddings):
        sim=cosine_similarity(faq_embedding,question_embedding)[0][0];
        if sim>max_sim:
            max_sim=sim;
            index_sim=index;
        if (sim>sec_max_sim and sec_max_sim != max_sim and index_sim!=index):
            sec_max_sim=sim;
            sec_index_sim=index;
    val1=(FAQdf.iloc[index_sim,1]).encode('unicode_escape')        
    val2=(FAQdf.iloc[sec_index_sim,1]).encode('unicode_escape')
    return ("SOLUTION 1" +"  " + str(val1) + " \n" + "SOLUTION 2" +"  " + str(val2))
  




def inforetrival(df,ques):
    cleaned_sentences=get_cleaned_sentences(df,stopwords=True)
    sent_embeddings=[]
    v2w_model=load_model()
    for sent in cleaned_sentences:
        sent_embeddings.append(getPhraseEmbedding(sent,v2w_model))
    question_text=ques
    question=clean_sentence(question_text,stopwords=False)
    question_embedding=getPhraseEmbedding(question,v2w_model)
    return retrieveAndPrintFAQAnswer(question_text,question_embedding,sent_embeddings,df,cleaned_sentences)
