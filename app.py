#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #Veri manipülasyonu veri ile ilgili
import numpy as np  #Matematik numarik kütüphane
import matplotlib.pyplot as plt #Veri görselleştirme
import seaborn as sns #Daha gelişmiş veri görselleştirme
import warnings
warnings.filterwarnings("ignore")


# # araba fiyatı tahmin eden model deployment

# In[5]:


df = pd.read_excel("cars.xls")
df


# In[6]:


#df.to_csv("cars.csv",index=False) #csv dosyasına çevirip kaydetme


# In[10]:


from sklearn.model_selection import train_test_split #Veri setinin %80'i egitim seti ve %20'i test seti olarak ayırma
from sklearn.linear_model import LinearRegression #Lineer Regresyon modeli
from sklearn.metrics import r2_score,mean_squared_error #Hata hesaplama
from sklearn.pipeline import Pipeline #Pipline oluşturma
from sklearn.preprocessing import StandardScaler,OneHotEncoder #Standartlasma
from sklearn.compose import ColumnTransformer #Veri setinin %80'i egitim seti ve %20'i test seti olarak ayırma


# In[12]:


X=df.drop("Price",axis=1) #Bagımsız degiskenler
y=df[["Price"]] #Bagımlı degisken


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) #Veri setinin %80'i egitim seti ve %20'i test seti olarak ayırma


# In[14]:


preproccer=ColumnTransformer(transformers=[('num',StandardScaler(),
                                           ['Mileage','Cylinder','Liter','Doors']),
                            ('cat',OneHotEncoder(),['Make','Model','Trim','Type'])])


# In[21]:


model=LinearRegression() #Lineer Regresyon modeli
pipe=Pipeline(steps=[('preproccer',preproccer),('model',model)]) #Pipline oluşturma

pipe.fit(X_train,y_train) #Pipline uygulama
y_pred=pipe.predict(X_test) #Tahmin


mean_squared_error(y_test,y_pred)**0.5,r2_score(y_test,y_pred) #Hata hesaplama


# In[25]:


import streamlit as st #Web uygulaması
def price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather):
	input_data=pd.DataFrame({
		'Make':[make],
		'Model':[model],
		'Trim':[trim],
		'Mileage':[mileage],
		'Type':[car_type],
		'Car_type':[car_type],
		'Cylinder':[cylinder],
		'Liter':[liter],
		'Doors':[doors],
		'Cruise':[cruise],
		'Sound':[sound],
		'Leather':[leather]
		})
	prediction=pipe.predict(input_data)[0]
	return prediction
st.title("Car Price Prediction :red_car: @Cem")
st.write("Enter Car Details to predict the price of the car")
make=st.selectbox("Make",df['Make'].unique())
model=st.selectbox("Model",df[df['Make']==make]['Model'].unique())
trim=st.selectbox("Trim",df[(df['Make']==make) & (df['Model']==model)]['Trim'].unique())
mileage=st.number_input("Mileage",200,60000)
car_type=st.selectbox("Type",df['Type'].unique())
cylinder=st.selectbox("Cylinder",df['Cylinder'].unique())
liter=st.number_input("Liter",1,6)
doors=st.selectbox("Doors",df['Doors'].unique())
cruise=st.radio("Cruise",[True,False])
sound=st.radio("Sound",[True,False])
leather=st.radio("Leather",[True,False])
if st.button("Predict"):
	pred=price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather)

	st.write("Predicted Price :red_car:  $",round(pred[0],2))


# In[ ]:




