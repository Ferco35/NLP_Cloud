import streamlit as st
import predictions
from predictions import *

text=st.text_input("Text to analyse")
nbrTopics=st.number_input('Nombre de topics')
if st.button('Detect'):
    st.write(text)
    st.write(nbrTopics)
    precision,topics=predictions.predict(text,int(nbrTopics))
    if precision > 0:
        st.warning("Polarité: " + str(precision) + " (COMMENTAIRE POSITIF)")
    else :
        st.warning("Polarité: " + str(precision) + " (COMMENTAIRE NEGATIF)")
        st.info("Topics :")
        st.info(topics)

else:
    st.write("Appuyer sur le Bouton Detect SVP")
