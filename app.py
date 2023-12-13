import streamlit as st
import pandas as pd
import numpy as np
from numpy import array
import pickle
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import GaussianNB
from streamlit_option_menu import option_menu

with st.sidebar :
    selected = option_menu ('Klasifikasi Berita',['Introduction', 'Implementation'],default_index=0)


if (selected == 'Introduction'):
    st.title('Deskripsi Singkat Mengenai Alur Sistem')
    st.write('Dalam sistem ini melalui beberapa tahapan diantaranya :')
    st.markdown('**1. Load Data**')
    st.write('Data diperoleh dari proses crawling pada berita online yaitu detik.com dan suararakyat.id, dan umlah data yang digunakan sebanyak 1046 dokumen')
    st.markdown('**2. Preprocessing**')
    st.write('Proses preprocessing ini meliputi : cleansing, casefolding, tokenizing, dan stopword')
    st.markdown('**3. Ekstraksi Fitur**')
    st.write('Dalam kasus kali ini, proses ekstraksi fitur menggunakan Term Frequency')
    st.markdown('**4. Pembagian Data**')
    st.write('Data setelah dilakukan proses ekstraksi fitur kemudian dilakukan proses splitting data dimana data training sebesar 80% dan data testing sebesar 20%')
    st.markdown('**5. Latent Dirichlet Allocation**')
    st.write('Data training yang sudah dihasilkan sebelumnya akan digunakan untuk melatih model lda dengan jumlah topik yang memiliki nilai akurasi terbaik yaitu 50 topik')
    st.markdown('**6. Modelling**')
    st.write('Hasil dari LDA ini akan dilatih ke dalam model yang selanjutnya digunakan untuk mengklasifikasikan data baru')


def preproses(inputan):
    clean_tag = re.sub('@\S+','', inputan)
    clean_url = re.sub('https?:\/\/.*[\r\n]*','', clean_tag)
    clean_hastag = re.sub('#\S+',' ', clean_url)
    clean_symbol = re.sub('[^a-zA-Z]',' ', clean_hastag)
    casefolding = clean_symbol.lower()
    token=word_tokenize(casefolding)
    listStopword = set(stopwords.words('indonesian')+stopwords.words('english'))
    stopword=[]
    for x in (token):
        if x not in listStopword:
            stopword.append(x)
    joinkata = ' '.join(stopword)
    return clean_symbol,casefolding,token,stopword,joinkata


if (selected == 'Implementation'):
    st.title("""Implementasi Data""")
    inputan = st.text_input('Masukkan Berita')
    submit = st.button("Submit")
    if submit :
        clean_symbol,casefolding,token,stopword,joinkata = preproses(inputan)

        with open('tf_uas.sav', 'rb') as file:
            vectorizer = pickle.load(file)
        
        hasil_tf = vectorizer.transform([joinkata])
        tf_name=vectorizer.get_feature_names_out()
        tf_array=hasil_tf.toarray()
        df_tf= pd.DataFrame(tf_array, columns = tf_name)

        with open('lda.sav', 'rb') as file:
            lda = pickle.load(file)

        hasil_lda=lda.transform(df_tf)   

        with open('naive.sav', 'rb') as file:
            naive = pickle.load(file)
        
        hasil_naive=naive.predict(hasil_lda)
        hasil =f"Berdasarkan data yang Anda masukkan, maka berita masuk dalam kategori  : {hasil_naive}"
        st.success(hasil)

        st.subheader('Preprocessing')
        st.markdown('**Cleansing :**')
        clean_symbol
        st.markdown('**Casefolding :**')
        casefolding
        st.markdown('**Tokenisasi :**')
        token
        st.markdown("**Stopword :**")
        stopword
        st.header("Term Frequency :")
        df_tf
        st.header("Latent Dirichlet Allocation :")
        hasil_lda