import streamlit as st
import pandas as pd

st.title('Machine Learning')
st.info('Ejercicio de entrenamiento')

with st.expander('Data'):
  st.write('**Raw Data:**')
  df = pd.read_csv('https://raw.githubusercontent.com/Jadonosouc1977/machine_learning/main/penguins_cleaned.csv')
  df
