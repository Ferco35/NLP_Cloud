import streamlit as st
import predictions
from predictions import *

text=st.text_input("Text to analyse")
nbrTopics=st.number_input('Nombre de topics')
if st.button('Detect'):
    st.write(text)
    st.write(nbrTopics)
    precision,topics=predictions.predict(text,nbrTopics)
    if(precision<0):
        st.alert("Polarité : ", precision ," - COMMENTAIRE NEGATIF")
        st.info(topics)
    else:
        st.alert("Polarité : ", precision, " - COMMENTAIRE POSITIF")

else:
    st.write("Appuyer sur le Bouton Detect SVP")