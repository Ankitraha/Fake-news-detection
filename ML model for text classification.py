#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt_alias
import seaborn as sns
import nltk 
import re
import wordcloud as wc


# In[3]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# In[4]:


real_df = pd.read_csv("real.csv")
fake_df = pd.read_csv("fake.csv")


# In[5]:


real_df.head()


# In[6]:


fake_df.head()


# In[7]:


real_df.shape, fake_df.shape


# In[8]:


fake_df['subject'].value_counts()


# In[9]:


plt_alias.figure(figsize=(10, 6))
sns.countplot(x = 'subject', data=fake_df)


# In[10]:


## Wordcloud
text = ' '.join(fake_df['text'].tolist()) # Coverting fake test data into list


# In[11]:


wordcloud = wc.WordCloud().generate(text)
from wordcloud import WordCloud
wordcloud = WordCloud().generate(text)
plt_alias.imshow(wordcloud)
plt_alias.axis('off')
plt_alias.tight_layout(pad=0)
plt_alias.show()


# In[12]:


#Exploring real news
real_df['subject'].value_counts()


# In[13]:


## Wordcloud
text = ' '.join(real_df['text'].tolist()) # Coverting real test data into list


# In[14]:


wordcloud = wc.WordCloud().generate(text)
from wordcloud import WordCloud
wordcloud = WordCloud().generate(text)
plt_alias.imshow(wordcloud)
plt_alias.axis('off')
plt_alias.tight_layout(pad=0)
plt_alias.show()


# In[15]:


real_df.sample(6)


# In[16]:


unknown_publishers = []
for index, row in enumerate(real_df.text.values):
    try:
        record = row.split(' - ', maxsplit=1)
        record[1]
        
        assert(len(record[0])<120)
    except:
        unknown_publishers.append(index)


# In[17]:


len(unknown_publishers)


# In[18]:


real_df.iloc[unknown_publishers]


# In[19]:


real_df.iloc[unknown_publishers].text


# In[20]:


real_df.iloc[8970]


# In[21]:


#real_df = real_df.drop(8970)


# In[22]:


publisher = []
tmp_text = []

for index, row in enumerate(real_df.text.values):
    if index in unknown_publishers:
        tmp_text.append(row)
        publisher.append('Unknown')
        
    else:
        record = row.split('-', maxsplit=1)
        publisher.append(record[0].strip())
        tmp_text.append(record[1].strip())


# In[23]:


real_df['publisher']= publisher
real_df['text'] = tmp_text


# In[24]:


empty_fake_df_index = [index for index,text in enumerate(fake_df.text.tolist()) if str(text).strip()==""]


# In[25]:


fake_df.iloc[empty_fake_df_index] # fake news data - empty 


# In[26]:


real_df['text'] = real_df['title'] + " " + real_df['text']
fake_df['text'] = fake_df['title'] + " " + fake_df['text']


# In[27]:


real_df['text'] = real_df['text'].apply(lambda x: str(x).lower())
fake_df['text'] = fake_df['text'].apply(lambda x: str(x).lower())


# In[28]:


#Preprocessing Text
real_df['class'] = 1
fake_df['class'] = 0


# In[29]:


real_df.columns


# In[30]:


real_df =real_df[['text', 'class']]


# In[31]:


fake_df =fake_df[['text', 'class']]


# In[32]:


data = pd.concat([real_df, fake_df], ignore_index=True)


# In[33]:


data.sample(10)


# In[34]:


import preprocess_kgptalkie as ps

data['text'] = data['text'].apply(lambda x: ps.remove_special_chars(x))


# In[35]:


#Vectorization --Word2vec
import gensim
y = data['class'].values


# In[36]:


X = [d.split() for d in data['text'].tolist()] # convert text data into list of list.


# In[37]:


type(X[0])


# In[38]:


print(X[0])


# In[39]:


DIM = 100 # these words will converted into sequence of 100 vectors.
w2v_model =  gensim.models.Word2Vec(sentences=X, vector_size=DIM, window=15, min_count=1)


# In[40]:


word_vectors = w2v_model.wv
vocab_size = len(word_vectors.index_to_key)
print("Number of words in the vocabulary:", vocab_size)


# In[41]:


w2v_model.wv['usa']


# In[42]:


w2v_model.wv.most_similar('india')


# In[43]:


# Initialize the Tokenizer
tokenizer = Tokenizer()
# Fit the Tokenizer on your text data
tokenizer.fit_on_texts(X)


# In[44]:


X = tokenizer.texts_to_sequences(X)


# In[45]:


X


# In[46]:


#tokenizer.word_index
#Analyze our text data which we currently have
plt_alias.hist([len(x) for x in X], bins = 700)
#Histogram data of total number of words present in our news.
plt_alias.show()


# In[47]:


nos = np.array([len(x) for x in X])
len(nos[nos>1000])


# In[48]:


maxlen= 1000
X = pad_sequences(X,maxlen=maxlen)# when sequence>1000 then truncated , when sequence<1000, the the 0 is added.


# In[49]:


len(X[150]) # we always getting thousand 


# In[50]:


vocab_size = len(tokenizer.word_index)+ 1
vocab = tokenizer.word_index

def get_weight_matrix(model):
    weight_matrix = np.zeros((vocab_size, DIM))
    
    for word, i in vocab.items():
         weight_matrix[i] = model.wv[word]
            
    return weight_matrix
        


# In[51]:


embedding_vectors = get_weight_matrix(w2v_model)


# In[52]:


embedding_vectors.shape


# In[53]:


model = Sequential()
model.add(Embedding(vocab_size, output_dim=DIM, weights = [embedding_vectors], input_length=maxlen, trainable=False))
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[54]:


model.summary()


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[56]:


model.fit(X_train, y_train, validation_split=0.3, epochs=1)


# In[57]:


#save the model
model.save('fake_news_detection_model.h5')


# In[58]:


y_pred = (model.predict(X_test) >=0.5).astype(int)


# In[59]:


accuracy_score(y_test, y_pred)


# In[60]:


print (classification_report(y_test, y_pred))


# In[61]:


def x(text):
    print("The input is text:", text)

def y(link):
    print("The input is a link:", link)

user_input = input("Enter text or link: ")

if user_input.startswith("http"):
    y(user_input)
else:
    x(user_input)


# In[64]:


from readability import Document
import requests
import re
from readability import Document
import requests
import re
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from bs4 import BeautifulSoup
def link_input(link):
    url = link
    # Send a request to the URL and retrieve the page content
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Create a Document object from the HTML content
        text = response.text
        cleaned_text = text.replace("&copy;", "")
        doc = Document(cleaned_text)

        # Extract the main text from the Document object
        main_text = doc.summary()
        # Clean up the main text by removing any unwanted characters
        main_text = re.sub(r'\n+', '\n', main_text)
        main_text = re.sub(r'\s+', ' ', main_text)
    
        # Remove any links present in the text
        main_text = re.sub(r'https?://[^\s]+', '', main_text)
        # Remove any texts that match the pattern "<p class="caption">...</p>"
        main_text = re.sub(r'<p class="caption">.*?</p>', '', main_text)
        # Remove any elements starting with "<" and ending with ">"
        main_text = re.sub("<.*?>", "", main_text)
        main_text_text = re.sub(r'\^ .*', '', main_text)


   


        # Extract the title of the web page
        title = doc.title()

        # Clean up the title by removing any unwanted characters
        title = re.sub(r'\n+', '\n', title)
        title = re.sub(r'\s+', ' ', title)
        

       # Print the main text and the title
        print("Title:", title)
        print("\nMain Text:\n", main_text)
    else:
        print("Failed to retrieve the page content")
        print("The input is a link:", link)
    x = [main_text]
    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=maxlen) 
    prediction=(model.predict(x) >=0.5).astype(int)
    print("Prediction:", prediction)
     
def text_input(text):
    print("The input is text:", text)
    x = [text]
    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=maxlen) 
    prediction=(model.predict(x) >=0.5).astype(int)
    print("Prediction:", prediction)
user_input = input("Enter text or link: ")

if user_input.startswith("http"):
    link_input(user_input)
else:
    text_input(user_input)
    
if user_input.startswith("http"):
    html_page = urllib.request.urlopen(user_input)
    soup = BeautifulSoup(html_page, 'html.parser')
    images = []
    for img in soup.find_all('img'):
        src = img.get('src')
        if src.endswith('.jpg'):
            src = "https:" + src
            images.append(src)

    for image_url in images:
        img_array = np.array(bytearray(urllib.request.urlopen(image_url).read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        plt.imshow(img)
        plt.show()
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

# Load the model
    model1 = load_model("keras_model.h5", compile=False)

# Load the labels
    class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
    real_image=0
    fake_image=0
    total_image=0
    for image_url in images:
        total_image+=1
        img_array = np.array(bytearray(urllib.request.urlopen(image_url).read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        image = Image.fromarray(img).convert("RGB")
        import PIL.Image
        if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
            PIL.Image.Resampling = PIL.Image

    # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
        image_array = np.asarray(image)

    # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
        data[0] = normalized_image_array

    # Predicts the model
        prediction = model1.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

    # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)
        if(index==0):
            real_image+=1
        else:
            fake_image+=1
    print("Total no. of images extracted:",total_image,"\n")
    print("No. of real images extracted:",real_image, "\n")
    print("No. of fake images extracted:",fake_image) 


# In[65]:


model5 = load_model("fake_news_detection_model.h5", compile=False)
a = ["Malaysia grew at the quickest pace in over two decades in 2022, making it the fastest-growing economy in Asia, as the pent-up demand helped the nation's GDP grow to 8.7 per cent, the highest level since 2000, data from the Bank Negara Malaysia and the Department of Statistics showed, Bloomberg reported."]
a = tokenizer.texts_to_sequences(a)
a = pad_sequences(a, maxlen=maxlen)
(model5.predict(a) >=0.5).astype(int)


# In[ ]:


x = ["Giving further details on the launch, US Air Force said the ICBM's test re-entry vehicle travelled approximately 4,200 miles to the Kwajalein Atoll in the Marshall Islands. It added that the trajectory displays the accuracy and reliability of the US ICBM systems."]
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=maxlen)
(model.predict(x) >=0.5).astype(int)


# In[66]:


x = ["The former commander of the NATO US forces in Europe 2014-2018, Gen. Ben Hodges, says the war could end in 2023 with air support to Ukraine."]
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=maxlen)
(model.predict(x) >=0.5).astype(int)


# In[ ]:




