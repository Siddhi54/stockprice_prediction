from bsedata.bse import BSE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64
#import our finalise model
import pickle
from bse_scrap import bse_scrape
import streamlit as st
from PIL import Image



bseScrap = bse_scrape()
Company_list = bseScrap.get_security_code()

st.title("Stock Price Prediction of BSE")
image = Image.open('bse.jpg')

st.image(image)

user_input = st.selectbox("Select company", Company_list )
seurity_code = bseScrap.get_securityCode_by_company(user_input)
#st.write(seurity_code)
data = bseScrap.get_databysecurity(seurity_code)

st.subheader('Data from 2022-2023')
st.write(data.describe())
#st.write(type(data))
#plot graph
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,7))
plt.plot(data.Close,'red')
#plt.show()
st.pyplot(fig)


#plot graph
st.subheader('Closing Price vs Time Chart with 10Days Moving Avg')
fig = plt.figure(figsize = (12,7))
plt.plot(data.ema,'black')
plt.plot(data.Close,'red')
#plt.show()
st.pyplot(fig)

r2_score_lst, Linearreg, Lassoreg, Ridgereg, regressor, ls, rr, x_test, y_test = bseScrap.train_test(data)

r2_score_maxindex = r2_score_lst.index(max(r2_score_lst))
result = r2_score_maxindex + 1

if result == 1:
    pipe = pickle.load(open("Linearmodel.pkl", 'rb'))
    predictor = pipe.predict(x_test)
    print('pred',predictor)
    pred = predictor.mean()
    st.subheader('Prediction vs Actual')
    fig =Linearreg.plot(figsize = (12,8), y=['y_test','y_pred_lr'], label =['Original price', 'Predicted price(lr)'])
    plt.ylabel('Price')
    plt.show()
    g = plt.savefig('op.png')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(g)
    st.subheader('Predicted Value')
    if st.button('Click here'):
        st.write('Predicted Value is:  ', str(pred))
    
   
if result == 2:
    pipe = pickle.load(open("Lassomodel.pkl", 'rb'))
    predictor = pipe.predict(x_test)
    print('pred',predictor)
    pred = predictor.mean()
    st.subheader('Prediction vs Actual')
    fig = Lassoreg.plot(figsize = (12,8),y=['y_test','y_pred_ls'], label =['Original price', 'Predicted price(ls)'])
    plt.ylabel('Price')
    plt.show()
    g = plt.savefig('op.png')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(g)
    
    st.subheader('Predicted Value')
    if st.button('Click here'):
        st.write('Predicted Value is:  ', str(pred))
    
    

if result == 3:
    pipe = pickle.load(open("Ridgemodel.pkl", 'rb'))
    predictor = pipe.predict(x_test)
    print('pred',predictor)
    pred = predictor.mean()
    st.subheader('Prediction vs Actual')
    #fig = plt.figure(figsize = (12,6))
    fig =Lassoreg.plot(figsize = (12,8),y=['y_test','y_pred_rr'], label =['Original price', 'Predicted price(rr)'])
    plt.ylabel('Price')
    plt.show()
    g = plt.savefig('op.png')
    st.pyplot(g)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('Predicted Value')
    if st.button('Click here'):
        st.write('Predicted Value is:  ', str(pred))

