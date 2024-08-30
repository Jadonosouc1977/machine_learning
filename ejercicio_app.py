import streamlit as st
import pandas as pd

st.title('Machine Learning')
st.info('Ejercicio de entrenamiento')

df = pd.read_csv('https://github.com/Jadonosouc1977/machine_learning/blob/main/penguins_cleaned.csv')
df
