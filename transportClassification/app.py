# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:02:59 2023

@author: qomondon8943@gmail.com
"""

import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform
plt = platform.systme()
if plt == 'Linnux': pathlib.Windows = pathlib.PosixPath
temp =  pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

#title
st.title('vehicle classification model,\n(only boat, car, airplane types)')

#upload a img
file= st.file_uploader('Upload a picture', type=['png','jpeg','gif','svg'])
if file:
    st.image(file)
    #PIL convertion
    img = PILImage.create(file)

    #model
    model = load_learner('transport_model.pkl')

    #prediction

    pred, pred_id, probs =  model.predict(img)
    st.success(f'prediction: {pred}')
    st.info(f'accuracy: {probs[pred_id]*100:.1f}%')

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab,)
    st.plotly_chart(fig)
