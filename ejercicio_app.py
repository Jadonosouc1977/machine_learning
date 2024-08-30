import streamlit as st
import pandas as pd
from sklearn.emsemble import RandomForestClassifier

st.title('Machine Learning')
st.info('Ejercicio de entrenamiento')

with st.expander('Data'):
  st.write('**Raw Data:**')
  df = pd.read_csv('https://raw.githubusercontent.com/Jadonosouc1977/machine_learning/main/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species',axis=1)
  X_raw

  st.write('**Y**')
  y_raw = df['species']
  y_raw
  
with st.expander('Data visualization:'):
  st.scatter_chart(data=df, x='bill_length_mm',y='body_mass_g',color = 'species')
  
# Input Features
with st.sidebar:
  st.header('Input Features :')
  island = st.selectbox('Island',('Biscoe','Dream','Torgensen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1,59.4,43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1,21.5,17.2)
  flipper_length = st.slider('Flipper length (mm)', 172.0,231.0,201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0,6300.0,4207.0)
  gender = st.selectbox('Gender',('male','female'))

  # Create a Dataframe for input features
  data = {'island':island,
        'bill_length_mm':bill_length_mm,
        'bill_depth_mm':bill_depth_mm,
        'flipper_length':flipper_length,
        'body_mass_g':body_mass_g,
        'sex':gender }

  input_df = pd.DataFrame(data,index=[0])
  input_penguins = pd.concat([input_df,X_raw],axis=0)


with st.expander('Input Features:'):
  st.write('**Input penguins**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins


# Data Preparation
# Encode X
encode = ['island','sex']
df_penguins = pd.get_dummies(input_penguins, prefix = encode)

X = df_penguins[1:]

input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie':0, 'Chinstrap':1,'Gentoo':2}

def target_encode(val):
  return target_mapper[val]
y = y_raw.apply(target_encode)
 

with st.expander('Data preparation:'):
  st.write('**Encoded (X) input penguis **')
  input_row
  st.write('**Encoded (y) **')
  y

# Model Traning and inference #
# Train the ML model #
clf = RandomForest()
clf.fit(X,y)

# Apply model to make predictions #
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

prediction_proba
