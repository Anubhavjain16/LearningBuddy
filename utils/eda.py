import streamlit as st

def show_summary(df):
    st.write("### 📊 Dataset Summary")
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isnull().sum().sum())
    st.write("Column Info:")
    st.write(df.describe(include="all"))
