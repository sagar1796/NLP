#!/usr/bin/env python
# coding: utf-8

# In[1]:


docs=[
    "recurrent neural Network",
    "Neural Network",
    "artificial Neural Network",
    
]


# In[2]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[4]:


tokenizer=Tokenizer(oov_token="<nothing>")


# In[5]:


tokenizer.fit_on_texts(docs)


# In[6]:


tokenizer.word_index


# In[7]:


tokenizer.word_counts


# In[8]:


tokenizer.document_count


# In[9]:


sequences=tokenizer.texts_to_sequences(docs)


# In[10]:


sequences


# In[12]:


from tensorflow.keras.utils import pad_sequences


# In[13]:


sequences=pad_sequences(sequences,padding="post")


# In[14]:


sequences


# In[16]:


#sentiment analysis on imdb dataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten


# In[17]:


(X_train,y_train),(X_test,y_test)=imdb.load_data()


# In[18]:


X_train


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


y_train


# In[22]:


print(len(X_train[0]))
print(len(X_train[1]))


# In[23]:


X_train=pad_sequences(X_train,padding="post",maxlen=50)
X_test=pad_sequences(X_test,padding="post",maxlen=50)


# In[26]:


print(len(X_train[0]))
print(len(X_train[1]))


# In[27]:


X_train.shape


# In[32]:


model=Sequential()
model.add(SimpleRNN(32,input_shape=(50,1),return_sequences=False)) #maxlen is 50 and 1 as output , return_sequence=False because it is a case of many to one
model.add(Dense(1,activation='sigmoid'))
model.summary()


# In[35]:


model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[36]:


model.fit(X_train,y_train,epochs=3,validation_data=(X_test,y_test))


# In[40]:


X_test[0].reshape(1,-1).shape


# In[41]:


predication=X_test[0].reshape(1,-1)


# In[46]:


result=model.predict(predication)[0][0]
if result > 0.5:
    print(f' positive review with accuracy {result}')


# In[47]:


##how to do encoding using keras embedding layer


# In[48]:


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer()
tokenizer.fit_on_texts(docs)


# In[51]:


sequences=tokenizer.texts_to_sequences(docs)
sequences


# In[52]:


from tensorflow.keras.utils import pad_sequences
sequences=pad_sequences(sequences,padding="post")
sequences


# In[54]:


from tensorflow.keras.layers import Embedding


# In[55]:


len(tokenizer.word_index)


# In[67]:


model=Sequential()
model.add(Embedding(4,output_dim=1,input_length=3))
model.summary()


# In[70]:


model.compile(optimizer="adam",metrics=["accuracy"])


# In[71]:


pred=model.predict(sequences)
print(pred)


# In[72]:


#sentiment analysis on imdb dataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding,SimpleRNN, Flatten


# In[73]:


(X_train,y_train),(X_test,y_test)=imdb.load_data()


# In[74]:


X_train=pad_sequences(X_train,padding="post",maxlen=50)
X_test=pad_sequences(X_test,padding="post",maxlen=50)


# In[77]:


model=Sequential()
#embedding 1000 is unique words and output dim is embedding dimens
model.add(Embedding(10000,output_dim=2, input_length=50))
model.add(SimpleRNN(32,input_shape=(50,1),return_sequences=False))
model.add(Dense(1,activation='sigmoid'))
model.summary()


# In[78]:


model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[80]:


model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))


# In[ ]:




