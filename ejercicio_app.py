import streamlit as st
import pandas as pd

st.title('Machine Learning')
st.info('Ejercicio de entrenamiento')

with st.expander('Data'):
  st.write('**Raw Data:**')
  df = pd.read_csv('https://raw.githubusercontent.com/Jadonosouc1977/machine_learning/main/penguins_cleaned.csv')
  df

  st.write('**X**')
  X = df.drop('species',axis=1)
  X

  st.write('**Y**')
  y = df['species']
  y
  
with st.expander('Data visualization:'):
  st.scatter_chart(data=df, x='bill_lenght_mm',y='body_mass',color = 'species')
  
