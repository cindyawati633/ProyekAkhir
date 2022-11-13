#!/usr/bin/env python
# coding: utf-8

# # BUSINESS UNDERSTANDING
# - **TOPIK: KTT G20**
# 
# - **Alasan memilih topik tersebut: Ingin mengetahui seberapa antusiasme rakyat Indonesia untuk menyambut KTT G20 di Bali**

# # DATA UNDERSTANDING

# ### 1. **Data Collection**

# In[2]:


import pandas as pd, numpy as np, matplotlib.pyplot as plt
import json, tweepy, requests, re,string
from requests_oauthlib import OAuth1
from tweepy import OAuthHandler
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from urllib.request import urlopen
from bs4 import BeautifulSoup


# In[3]:


with open("token.json")as f:
  tokens = json.load(f)

bearer_token = tokens['bearer_token']
api_key = tokens['api_key']
api_key_secret = tokens['api_key_secret']
access_token = tokens['access_token']
access_token_secret = tokens['access_token_secret']
tokens.keys()


# In[4]:


auth = tweepy.OAuthHandler(api_key,api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)


# In[5]:


username= "G20"
max_result=19
posts=tweepy.Cursor(api.search_tweets,q=username,lang='id',tweet_mode='extended',).items(max_result)


# In[6]:


df=pd.DataFrame([tweet.full_text for tweet in posts],columns=['Tweets'])
df.head(19)


# In[7]:


alamat="https://id.wikipedia.org/wiki/G20"
html = urlopen(alamat)
data = BeautifulSoup(html, 'html.parser')
table = data.findAll("table", {"class":"wikitable"})


# In[8]:


table = data.findAll("table", {"class":"wikitable"})[0]
rows = table.findAll("tr")


# In[9]:


table = data.findAll("table", {"class":"wikitable"})[0]
rows = table.findAll("tr")
for row in rows:
    for cell in row.findAll(["td", "th"]):
         print(cell.get_text())


# In[10]:


hasil = []
for row in rows:
    info = []
    for cell in row.findAll(["td", "th"]):
        info.append(cell.get_text())
    hasil.append(info)


# In[11]:


df1 = pd.DataFrame(hasil, columns =['Tahun','Ke','Tanggal', 'Lokasi','Pemimpin tuan rumah','Situs web'])
display(df1)


# In[12]:


df12=df1.drop(labels=[0,0], axis=0)
display(df12)


# In[13]:


df12.reset_index(drop = True, inplace = True)
display(df12)


# In[14]:


df13= df12.drop(['Ke',"Situs web"],axis=1)


# In[15]:


df13


# In[16]:


dfbaru=pd.DataFrame(df13)


# In[17]:


dfbaru


# In[18]:


df


# In[19]:


result = pd.concat([df13, df], axis=1, join='inner')
display(result)


# In[20]:


result.to_csv('datagabungan.csv')


# In[21]:


dfresult=pd.read_csv('datagabungan.csv')
dfresult.head()


# ### 2. **Data Processing**

# In[43]:



def clean_lower(lwr):
    lwr = lwr.lower() 
    return lwr
# Buat kolom tambahan untuk data description yang telah dicasefolding  
dfresult['lwr'] = dfresult['Tweets'].apply(clean_lower)
casefolding=pd.DataFrame(dfresult['lwr'])
casefolding

#Remove Puncutuation
clean_spcl = re.compile('[/(){}\[\]\|@,-.]')
clean_symbol = re.compile('[^0-9a-z]')
def clean_punct(text):
    text = clean_spcl.sub('', text)
    text = clean_symbol.sub(' ', text)
    return text
# Buat kolom tambahan untuk data description yang telah diremovepunctuation   
dfresult['clean_punct'] = dfresult['lwr'].apply(clean_punct)
dfresult['clean_punct']

#whitespace
def _normalize_whitespace(text):
    corrected = str(text)
    corrected = re.sub(r"//t",r"\t", corrected)
    corrected = re.sub(r"( )\1+",r"\1", corrected)
    corrected = re.sub(r"(\n)\1+",r"\1", corrected)
    corrected = re.sub(r"(\r)\1+",r"\1", corrected)
    corrected = re.sub(r"(\t)\1+",r"\1", corrected)
    return corrected.strip(" ")
dfresult['clean_double_ws'] = dfresult['clean_punct'].apply(_normalize_whitespace)
dfresult['clean_double_ws']

#remove numbers
def clean_number(nmbr):
    nmbr=re.sub(r"\d+", "",nmbr)
    return nmbr
dfresult['clean_number'] = dfresult['clean_double_ws'].apply(clean_number)
dfresult['clean_number']

def case_folding2(data):
    #data = data.lower()
    data = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",data).split())
    data = re.sub(r"\d+", "", data)
    data = data.translate(str.maketrans("","",string.punctuation))
    data = re.sub(r"\n","",data)
    data = re.sub(r"\t","",data)
    data=re.sub('#','',data)
    data=re.sub('rt(RT)[\s]+','',data)
    data=re.sub('https\\:\/\/\S+','',data)
    return data
dfresult['clean_all'] = dfresult['clean_number'].apply(case_folding2)
dfresult['clean_all']


# In[44]:


display(dfresult)


# In[45]:


stopword = set(stopwords.words('indonesian'))
def clean_stopwords(text):
    text = ' '.join(word for word in text.split() if word not in stopword) # hapus stopword dari kolom deskripsi
    return text
# Buat kolom tambahan untuk data description yang telah distopwordsremoval   
dfresult['clean_sw'] = dfresult['clean_all'].apply(clean_stopwords)


# In[46]:


stopwords.words('indonesian')
[
 'bali',
    'KTTG20'
]

add = pd.DataFrame(dfresult['clean_sw'])
dfresult['add_swr']= add.replace(to_replace =['bali','KTTG20'],  
                            value ="", regex= True) 
dfresult['add_swr']


# In[47]:


ps = nltk.PorterStemmer()
print(ps.stem('KTTG20'))

def porterstemmer(text):
  text = ' '.join(ps.stem(word) for word in text.split() if word in text)
  return text
# Buat kolom tambahan untuk data description yang telah dilemmatization   
dfresult['desc_clean_porterstem'] = dfresult['add_swr'].apply(porterstemmer)
dfresult['desc_clean_porterstem']


# In[48]:


df_baru = dfresult.drop(dfresult.columns[[0, 6, 7,8,9,10,11,12]], axis=1)
df_baru


# In[49]:


bin_range = np.arange(0, 260, 10)
dfresult['desc_clean_porterstem'].str.len().hist(bins=bin_range)
plt.show()
 


# ### 3. Data Analisis

# In[50]:


bin_range = np.arange(0, 50)
dfresult['desc_clean_porterstem'].str.split().map(lambda x: len(x)).hist(bins=bin_range)
plt.show()


# In[51]:


dfresult['desc_clean_porterstem'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
plt.show()


# In[52]:


dfresult['desc_clean_porterstem'] = dfresult['desc_clean_porterstem'].apply(lambda x: word_tokenize(str(x)))


# In[53]:


tweets = [word for tweet in dfresult['Tweets'] for word in tweet]
fqdist = FreqDist(tweets)

print(fqdist)


# In[54]:


most_common_word = fqdist.most_common(20)

print(most_common_word)


# In[55]:


fqdist.plot(20,cumulative=False)

plt.show()


# In[56]:


result1 = pd.Series(nltk.ngrams(tweets, 2)).value_counts()[:20]


# In[57]:


print (result1)


# In[60]:


df_baru.to_csv('datagabungan2.csv')


# In[59]:


import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from numpy import array
import numpy as np
import tqdm


# In[61]:


df_data2 = pd.read_csv("datagabungan2.csv")

df_data2


# In[63]:


df_data3 = [datagabungan2.split() for datagabungan2 in df_data2["desc_clean_porterstem"]]
df_data3


# In[66]:


dictionary = corpora.Dictionary(df_data3)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in df_data3]


# In[73]:


Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(
    random_state=100,
    chunksize=100,
    per_word_topics=True,
    corpus = doc_term_matrix, 
    num_topics=3, 
    id2word = dictionary, 
    passes=50
    )


# In[74]:


from pprint import pprint
pprint(ldamodel.print_topics())


# In[75]:


pip install pyLDAvis


# In[76]:


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


# In[77]:


pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim_models.prepare(ldamodel, doc_term_matrix, dictionary)
LDAvis_prepared
#pyLDAvis.display(vis)


# In[72]:


import streamlit as st 


# # REPORTING

# ### 1. **Analisis**

# Dari hasil pencarian di twitter dan wikipedia, berdasarkan grafik pyLDA didapatkan kata yang paling banyak adalah presiden, ani,
# anies baswedan, dan hadir.
# 
# Kemungkinan topik yang relevan presiden yang hadir dalam KTT G20.
# 
# Adapun analisis ini terdapat keanehan seperti :
# 
# 1. Dalam proses cleaning data masih terdapat kata-kata yang tidak diperlukan (Contoh:rt,https, tanda baca ", tanda baca ,)
# 2. Dalam grafil LDA kata yang paling banyak muncul adalah ani, mungkin kata ani ini merujuk pada Anies Baswedan
# 3. Proses cleaning data masih belum sempurna karena masih banyak kata-kata yang mirip namun dianggap kata berbeda sehingga membuat pemahaman yang berbeda

# ### 2. **Kesimpulan**
# 
# Dari proses analisis ini masih terdapat kekurangan terutama dalam proses cleaning data yang belum maksimal. Hal tersebut dibuktikan pada kata-kata yang kurang relevan yang muncul pada grafik LDA.

# In[ ]:




