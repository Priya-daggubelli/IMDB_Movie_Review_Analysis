import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model = pk.load(open('model.pk1','rb'))
scaler = pk.load(open('scaler.pk1','rb'))
review = st.text_input("Enter the movie review")

if st.button("Predict"):
    review_scale = scaler.transform([review]).toarray()
    result=model.predict(review_scale)
    if result[0] == 0:
        st.write("Negative experience")
    else:
        st.write("Positive experience")

    