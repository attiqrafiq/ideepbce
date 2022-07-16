import streamlit as st
# from streamlit import report_session
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict
# import protFeature
import pickle
import requests
from io import BytesIO
from fastai.text.all import *
# import pandas as pd
# import numpy as np
# from propy import PyPro
# # from tensorflow.keras.models import model_from_json

# # ################## Preparation for Web App
# with open("epi__DPC_Scale.pkl", 'rb') as file:
#     std_scale = pickle.load(file)
std_scale=pickle.load(open('epi__DPC_Scale.pkl','rb'))
# _Clf=pickle.load(open('./epi_all_Model.pkl','rb'))

epi_length = 20
st.title("iLBCE-Deep")

# download model from Dropbox, cache it and load the model into the app
@st.cache(allow_output_mutation=True)
def load_model(url):
    modelLink = url
    model = requests.get(modelLink).content
    return model


modelFile = load_model("https://raminay.com/model/epi__DPC_Model.pkl")
model = BytesIO(modelFile)
learn_inf = load_learner(model)

def SimpleFastaParser(fasta_sequence):
    seq = fasta_sequence.split('\n')
    seq = seq[1:]
    re = ''
    for x in seq:
        re = re + x[:len(x)]
    return re


def SimpleParser(sequence):
    seq = sequence.split('\n')
    re = ''
    for x in seq:
        re = re + x[:len(x)]
    return re

html_head = """
    ## iLBCE-Deep: A Deep Learning Model based Server to Identify BCell Epitopes
    BCells epitopes are immunogenic, having the capability to bind with the antigens to create antibodies to stimulate 
    the bodies auto-immune system to fight against the pathogens.
    iLBCE-Deep server is based on the Deep Convolutional Neural Network  (CNN) Model developed to predict linear bcell epitopes 
    from a protein sequence. This proposed prediction model identify the epitopes with increased accuracy.
    """
st.markdown(html_head, unsafe_allow_html=True)
# st.header("*Protein Sequence*")
st.image("archi_img.png", caption="Proposed methodology to develop iLBCE-Deep Classification Model", use_column_width=True)
st.sidebar.header("*Protein Sequence*")
seq = st.sidebar.text_area("Input a sequence in fasta format or in simple text", height=200)

# epi_length = st.sidebar.number_input("Enter the length of an epitope to predict",
#                                      min_value=6, max_value=49, value=19, step=1)

epi_length = st.sidebar.slider("Enter the length of an epitope to predict",
                                     min_value=6, max_value=49, value=19, step=1)
if epi_length:
    st.session_state.load_state = True
st.subheader('Click the Button for Sample Sequence or Download the Complete Data')
col1, col2 = st.columns([1, 1])
with col1:
    btnExample = st.button('Sample Sequence')
# initialize session state
if "load_state" not in st.session_state:
    st.session_state.load_state = False

if btnExample:
    st.session_state.load_state = True
    st.info("Example Protein Sequences")
    st.code(
        'MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELSKDIGSESTEDQAMEDIKQMEAESISSSEEIVPNSVEQKHIQKEDVPSERYLGYLEQLLRLKKYKVPQLEIVPNSAEERLHSMKEGIHAQQKEPMIGVNQELAYFYPELFRQFYQLDAYPSGAWYYVPLGTQYTDAPSFSDIPNPIGSENSEKTTMPLW', language="markdown")
    st.code(
        'MASQKRPSQRHGSKYLATASTMDHARHGFLPRHRDTGILDSIGRFFGGDRGAPKRGSGKDSHHPARTAHYGSLPQKSHGRTQDENPVVHFFKNIVTPRTPPPSQGKGRGLSLSRFSWGAEGQRPGFGYGGRASDYKSAHKGFKGVDAQGTLSKIFKLGGRDSRSGSPMARR', language="markdown")
    st.code(
        'MASMQHFSLAALLLAASICLGDADRTECQLPLDKGTPCTQEGGVKPSVAWWHDDKSGICLSFKYTGCGGNANRFTTIKNCEQHCKMPDRGACALGKKPAEDSNGEQLVCAGMREDKCPNGYQCKMMAFMGLCCPTKEEELFAREYEGVCKSGKPVKMDRGSGWMMTILGKSCDDQFCPEDAKCERGKLFANCCK', language="markdown")
#with col2:
#    with open('deepdpc.zip', 'rb') as f:
#       st.download_button('Download Zip', f, file_name='archive.zip')  # Defaults to 'application/octet-stream'


if st.sidebar.button('Submit'):
    st.session_state.load_state = True
    allowed_chars = set('ACDEFGHIKLMNPQRSTVWXY')
    # import extractDPCFeatures
    
    if seq == "":
        st.error("Please input the sequence first")
        exit(code=None)
    if not set(seq).issubset(allowed_chars):
        st.error("Invalid protein sequence, please provide correct protein sequence")
        exit(code=None)
    if '>' in seq:
        sequence = SimpleFastaParser(seq)
    else:
        sequence = SimpleParser(seq)
    result, seq, seqnew = protFeature.feature_result(sequence, epi_length)
    st.info("Entered Protein Sequences for BCell Epitope Prediction")
    st.write(seq)
    st.info("Here "+"B"+" Represent the BCell Epitopes found in Protein Sequences")
    st.write(seqnew)
    st.info("Identified BCell Epitopes in Protein Sequences with Site Index")
    st.table(result)



# streamlit run ideepDPCBCE.py
