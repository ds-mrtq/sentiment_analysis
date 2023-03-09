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

from io import BytesIO
from streamlit_chat import message
import streamlit as st


# Functions
# function to download dataframe as CSV file
def download_df(df):
    output_buffer = BytesIO()
    # write the DataFrame to bytes using UTF-8 encoding
    df.to_csv(output_buffer, encoding='utf-8', index=False)
    # get the value of the buffer
    output_bytes = output_buffer.getvalue()
    
    return output_bytes


# Load the saved pipeline using joblib
loaded_pipeline = load('sentiment_analysis_best_pipeline.joblib')

# Load sample data
# data = pd.read_csv('data/Pre_Products_Shopee_comments.csv').sample(n=10)[['rating','comment', 'sentiment']]
# data
# data = data['comment']
# data.to_csv("sample_data.csv", encoding='utf-8', index=False, header=False)
# y_pred = loaded_pipeline.predict(data)   
# y_pred

st.title('Sentiment Classification')


st.sidebar.title('Sentiment Classification by file uploaded')
st.sidebar.write("""
         #### Upload file to classify 
         """)

flag = False
lines = None

uploaded_file_1 = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file_1 is not None:
    lines = pd.read_csv(uploaded_file_1, header=None)
    # st.dataframe(lines)
    # st.write(lines.columns)
    lines = lines[0]     
    flag = True  

if flag:
    if len(lines)>0:
        st.write("Content:")
        # st.dataframe(lines) 
        input_df = pd.DataFrame(lines)
        input_df = input_df.rename(columns={input_df.columns[0]: 'comment'})
        input_df
        
        predicted_labels = loaded_pipeline.predict(lines)  
        predited_df = pd.DataFrame(predicted_labels, columns=['predict'])  
        
        st.write("Prediction:")
        result_df = pd.concat([input_df,predited_df], axis=1)
        result_df['predict_sentiment'] = result_df['predict'].apply(lambda x: 'Tích cực' if x > 0 else 'Tiêu cực')
        result_df = result_df[['comment', 'predict_sentiment']]
        result_df

        # create the download link 
        st.download_button(label="Download Result", data=download_df(result_df), file_name='result.csv', mime='text/csv')

    flag = False
