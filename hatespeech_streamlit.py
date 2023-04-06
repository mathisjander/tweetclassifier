
# This file contains the streamlit application hosting the hate speech classifier model
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from tokenizer import tokenize

# unpickle tfidf-classifier pipeline
with open('pkl_objects/pipe_lr.pkl', 'rb') as f:
    pipe = pickle.load(f)

# define UI
st.title('Hate Speech Classification')

tweet = st.text_input('Please enter your tweet to check for hate speech')

# prediction based on input
prediction = pipe.predict([tweet])
proba = pipe.predict_proba([tweet])

label = {0:'No hate speech', 1:'Hate Speech'}

if prediction == 1:
    certainty = proba[0][1]
else:
    certainty = proba[0][0]

# returning prediction to UI
st.write('Prediction: ' + str(label[prediction[0]]))
st.write('Certainty: ' + str(round(certainty*100, 1)) + '%')