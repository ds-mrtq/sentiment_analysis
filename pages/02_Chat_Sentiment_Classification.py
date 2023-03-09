import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle

import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from joblib import dump, load

import openai
from streamlit_chat import message
import streamlit as st

# Load the saved pipeline using joblib
loaded_pipeline = load('sentiment_analysis_best_pipeline.joblib')
# # Make predictions on new data
# new_data = ['Cái áo này đẹp quá', 'Cái áo này quá xấu', 'Cái áo này bình thường']
# predicted_labels = loaded_pipeline.predict(new_data)
# predicted_labels[0]

openai.api_key=st.secrets["apikey"]



#This function utilizes the OpenAI Completion API to generate a response based on the given prompt. 
#The temperature setting of the API affects how random the response is. 
#A higher temperature will generate more unpredictable responses while a
#lower temperature will lead to more predictable ones.
# max_tokens: length of respond
@st.cache_data
def generate_response(prompt):
    completions = openai.Completion.create (
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.6,
        max_tokens=200, 
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )

    message = completions.choices[0].text
    return message

st.title('ChatBot 🤖 🤖')


st.sidebar.title('ChatBot  🤖 🤖')
st.sidebar.write("""
         #### Try my chatbot made with my sentiment classify model (Logistic Regreesion), openAI, GPT-3 and  Streamlit. 
         """)

if 'sentiment_classify' not in st.session_state:
    st.session_state['sentiment_classify'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    #input_text = st.text_input("Human [enter your message here]: "," Hello Mr AI how was your day today? ", key="input")
    input_text= st.text_input('Human [enter your message here]:', '')
    return input_text 


user_input = get_text()



def sentiment_classify(loaded_pipeline, user_input):
    predicted_labels = loaded_pipeline.predict([user_input])
    sentiment_output = ''
    if predicted_labels[0] > 0:
        sentiment_output = '🤖 💭 👉 Đánh giá bình luận này là: Tích cực 😄'
    else:
        sentiment_output = '🤖 💭 👉 Đánh giá bình luận này là: Tiêu cực 😡'
    st.session_state.sentiment_classify.append(sentiment_output)

if user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)    
    st.session_state.generated.append(output)
    sentiment_classify(loaded_pipeline, user_input)
    



if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["sentiment_classify"][i], key=str(i) + '_sentiment_classify')
        message(st.session_state["generated"][i], key=str(i))
        
