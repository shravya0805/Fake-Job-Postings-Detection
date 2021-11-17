import streamlit as st
import pandas as pd
def app():
    st.image("https://www.surveysensum.com/wp-content/uploads/2020/02/SENTIMENT-09-1.png", width=400)
    st.markdown("<h1 style='text-align: center; color: #f6c453';>Sentimental Analysis</h1>", unsafe_allow_html=True)  
    st.write("""## Women's clothing E-commerce Reviews Dataset""")
    Reviews = pd.read_csv(r'C:\Users\shrav\Downloads\Reviews.csv')
    st.dataframe(Reviews)