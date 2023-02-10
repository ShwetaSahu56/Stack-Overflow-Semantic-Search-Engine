#!/usr/bin/env python
# coding: utf-8

# **Search Questions similar to query based on their Cosine-similarity value**

# In[1]:


import pandas as pd
import numpy as np
import pickle
import sklearn
import streamlit as st


# In[ ]:


st.title('Stack Overflow Search Engine')
st.markdown('Currenty working for limited Questions related to Javascript, java and c# only')
st.header('Enter your query')
query = st.text_input('Search Query')
st.button('Search', key='srchBtn')
if st.button('Say hello'):
    result = GetSimilarQuestions(query, 10)
    st.dataframe(result)


# **Function to get k-most similar questions to the query string based on cosine similarity**

# In[5]:


def GetSimilarQuestions(query, no_sim_ques):
    ''' This function finds k-most similar questions to the searched query based on their cosine similarity values'''
    
    #decontractions
    query = query.replace("won't", "will not").replace("can\'t", "can not").replace("n\'t", " not").replace("\'re", " are").replace("\'s", " is").replace("\'d", " would").replace("\'ll", " will").replace("\'t", " not").replace("\'ve", " have").replace("\'m", " am")

    query=re.sub(r'[^A-Za-z0-9#+\-]+',' ',query)
    print(query)
    
    #LSTM model based word-vector dictionary and Question Embeddings
    word_dict = pickle.load(open("Best_LSTM_Model_Vocab_Vector_dict.pkl", "rb"))
    lstm_embedded_questions = pd.read_pickle("Questions_lstm_embeddings_dataset.pkl")
    tfidf_wtd_ques_vec_embeddings = np.array(lstm_embedded_questions['tfidf_wtd_lstm_embed_questions'].tolist())
    
    #get tf-idf vectorizer model
    questions_tfidf = pickle.load(open("tfidf_model_ques.pickle", "rb"))

    # we are converting a dictionary with word as a key, and the idf as a value
    idf_dict = dict(zip(questions_tfidf.get_feature_names_out(), list(questions_tfidf.idf_)))
    tfidf_vocab = set(questions_tfidf.get_feature_names_out())
    
    #convert query words to vectors if word is present in both model vocabulary and tf-idf vectorizer vocabulary 
    query_vector = np.array([word_dict[w] for w in query.split() if w in word_dict and w in tfidf_vocab])
    
    #get tf-idf valuesfor each query words if word is present in both model vocabulary and tf-idf vectorizer vocabulary
    query_tf_idf = np.array([(idf_dict[w]*(query.count(w)/len(query.split()))) for w in query.split() if w in word_dict and w in tfidf_vocab ])
    
    #get tf-idf weighted vector embedding for query string
    query_tf_idf_vec = np.sum(query_vector*query_tf_idf[:,None], axis=0)
    if(np.sum(query_tf_idf)!=0):
        query_tfidf_wtd_vec = query_tf_idf_vec/np.sum(query_tf_idf)
  
    #get cosine similiarity value of each question w.r.t. query string
    cos_sim = pd.Series(sklearn.metrics.pairwise.cosine_similarity(query_tfidf_wtd_vec.reshape(1, -1), tfidf_wtd_ques_vec_embeddings)[0])
    
    #get n-larget values i.e. k (no_sim_ques) most similar questions to query string
    sim_questions = cos_sim.nlargest(no_sim_ques).index
    sim_ques_id_title =  lstm_embedded_questions.iloc[sim_questions][['Id', 'Title']]
    return sim_ques_id_title


# In[ ]:




