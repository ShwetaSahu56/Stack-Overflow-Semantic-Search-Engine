#!/usr/bin/env python
# coding: utf-8

# **Search Questions similar to query based on their Cosine-similarity value**

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit as st

def make_clickable(ques_id, title):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = title
    link = "https://stackoverflow.com/questions/"+str(ques_id)
    return f'<a target="_blank" href="{link}">{text}</a>'

@st.cache
def get_model_files():
    #LSTM model based word-vector dictionary and Question Embeddings
    word_dict = pickle.load(open("Best_LSTM_Model_Vocab_Vector_dict.pkl", "rb"))
    lstm_embedded_questions = pd.read_pickle("Questions_lstm_embeddings_dataset.pkl")
    tfidf_wtd_ques_vec_embeddings = np.array(lstm_embedded_questions['tfidf_wtd_lstm_embed_questions'].tolist())
    #get tf-idf vectorizer model
    questions_tfidf = pickle.load(open("tfidf_model_ques.pickle", "rb"))
    # we are converting a dictionary with word as a key, and the idf as a value
    idf_dict = dict(zip(questions_tfidf.get_feature_names_out(), list(questions_tfidf.idf_)))
    tfidf_vocab = set(questions_tfidf.get_feature_names_out())
    return word_dict, lstm_embedded_questions, tfidf_wtd_ques_vec_embeddings, idf_dict, tfidf_vocab

def GetSimilarQuestions(query, no_sim_ques):
    ''' This function finds k-most similar questions to the searched query based on their cosine similarity values'''
    
    #decontractions
    query = query.replace("won't", "will not").replace("can\'t", "can not").replace("n\'t", " not").replace("\'re", " are").replace("\'s", " is").replace("\'d", " would").replace("\'ll", " will").replace("\'t", " not").replace("\'ve", " have").replace("\'m", " am")

    query=re.sub(r'[^A-Za-z0-9#+]+',' ',query)
    
    #LSTM model based word-vector dictionary and Question Embeddings
    #word_dict = pickle.load(open("Best_LSTM_Model_Vocab_Vector_dict.pkl", "rb"))
    #lstm_embedded_questions = pd.read_pickle("Questions_lstm_embeddings_dataset.pkl")
    #tfidf_wtd_ques_vec_embeddings = np.array(lstm_embedded_questions['tfidf_wtd_lstm_embed_questions'].tolist())
    
    #get tf-idf vectorizer model
    #questions_tfidf = pickle.load(open("tfidf_model_ques.pickle", "rb"))

    # we are converting a dictionary with word as a key, and the idf as a value
    #idf_dict = dict(zip(questions_tfidf.get_feature_names_out(), list(questions_tfidf.idf_)))
    #tfidf_vocab = set(questions_tfidf.get_feature_names_out())
    
    word_dict, lstm_embedded_questions, tfidf_wtd_ques_vec_embeddings, idf_dict, tfidf_vocab = get_model_files()
    
    #convert query words to vectors if word is present in both model vocabulary and tf-idf vectorizer vocabulary 
    query_vector = np.array([word_dict[w] for w in query.split() if w in word_dict and w in tfidf_vocab])
    
    #get tf-idf valuesfor each query words if word is present in both model vocabulary and tf-idf vectorizer vocabulary
    query_tf_idf = np.array([(idf_dict[w]*(query.count(w)/len(query.split()))) for w in query.split() if w in word_dict and w in tfidf_vocab ])
    
    #get tf-idf weighted vector embedding for query string
    query_tf_idf_vec = np.sum(query_vector*query_tf_idf[:,None], axis=0)
    if(np.sum(query_tf_idf)!=0):
        query_tfidf_wtd_vec = query_tf_idf_vec/np.sum(query_tf_idf)
  
    #get cosine similiarity value of each question w.r.t. query string
    cos_sim = pd.Series(cosine_similarity(query_tfidf_wtd_vec.reshape(1, -1), tfidf_wtd_ques_vec_embeddings)[0])
    
    #get n-larget values i.e. k (no_sim_ques) most similar questions to query string
    sim_questions = cos_sim.nlargest(no_sim_ques).index
    sim_ques_id_title =  lstm_embedded_questions.iloc[sim_questions][['Id', 'Title']]
    sim_ques_id_title['Similar Questions'] = sim_ques_id_title.apply(lambda x: make_clickable(x['Id'], x['Title']), axis=1)
    df = sim_ques_id_title[['Similar Questions']]
    return df


st.title('Stack Overflow Search Engine')
st.markdown('Currenty working for limited Questions related to Javascript, Java and C# only')
st.header('Enter your query')
query = st.text_input('Search Query')
#st.button('Search', key='srchBtn')
# pandas display options
pd.set_option('display.max_colwidth', -1)


# show data
if st.button('Search'):
    result = GetSimilarQuestions(query, 10)
    st.write(result.to_html(index = False, escape = False, render_links = True), unsafe_allow_html = True)
    #st.dataframe(result)




