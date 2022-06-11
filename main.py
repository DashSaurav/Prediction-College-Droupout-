import pickle
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
header= st.sidebar.container()

with header:
    padding = 0
    img = Image.open("MicrosoftTeams-image.png")
    st.image(img, width=300)

def user_input_features():
    c = st.columns(3)
    with c[0]:
        Residence_city = st.selectbox('Residence City',('LOCAL','NEIGHBOR','FOREIGN'))
    with c[1]:
        Socioeconomic_level = st.slider('Select Socioeconomic Level',-1,2,1)
    with c[2]:
        Civil_status = st.selectbox('Civil Status',('Unmarried', 'Married', 'Free union', 'Separated'))
    cc = st.columns(3)
    with cc[0]:
        Age = st.slider('Select Age',18,70,23)
    with cc[1]:
        State = st.selectbox('Select State',('LOCAL','NEIGHBOR','FOREIGN'))
    with cc[2]:
        Province = st.selectbox('Select Province',('LOCAL','NEIGHBOR','FOREIGN'))
    ccc = st.columns(3)
    with ccc[0]:
        Desired_program = st.selectbox('Select a Program You are in', ("INFORMATIC ENGINEERING","ELECTRONIC AUTOMATION TECHNOLOGY","UNSPECIFIED"))
    with ccc[1]:
        Family_income = st.slider('Select Family Income',0,10000000,1200000)
    with ccc[2]:
        Father_level = st.selectbox('Select Fathers Study Level', ("UNDERGRADUATE","HIGH SCHOOL","PRIMARY SCHOOL","TECHNICAL","TECHNOLOGIST","UNREGISTERED"))
    cccc = st.columns(3)
    with cccc[0]:
        Mother_level = st.selectbox('Select Mothers Study Level', ("UNDERGRADUATE","HIGH SCHOOL","PRIMARY SCHOOL","TECHNICAL","TECHNOLOGIST","UNREGISTERED"))
    with cccc[1]:
        STEM_subjects = st.slider('STEM Subject Marks', 10.0,100.0, 43.0)
    with cccc[2]:
        H_subjects = st.slider('H subjects', 10.0,100.0, 45.2)

    data = {'Residence_city':[Residence_city],'Socioeconomic_level':[Socioeconomic_level],'Age':[Age], 
            'Civil_status':[Civil_status],'State':[State],'Province':[Province],'Desired_program':[Desired_program],
            'Family_income':[Family_income],'Father_level':[Father_level],'Mother_level':[Mother_level],
            'STEM_subjects':[STEM_subjects],'H_subjects':[H_subjects],
            }
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()
# st.write(input_df)

churn_raw = pd.read_csv("rulesDataset.csv")
churn_raw.fillna(0, inplace=True)
churn = churn_raw.drop(columns=['Dropout'])
df = pd.concat([input_df,churn],axis=0)

encode = ['Residence_city','Civil_status','State','Province','Desired_program','Father_level','Mother_level']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)
df.fillna(0, inplace=True)

features = ['Socioeconomic_level', 'Age', 'Vulnerable_group', 'Family_income',
       'STEM_subjects', 'H_subjects', 'Residence_city_FOREIGN',
       'Residence_city_LOCAL', 'Residence_city_NEIGHBOR',
       'Civil_status_Free union', 'Civil_status_Married',
       'Civil_status_Separated', 'Civil_status_Unmarried', 'State_FOREIGN',
       'State_LOCAL', 'State_NEIGHBOR', 'Province_FOREIGN', 'Province_LOCAL',
       'Province_NEIGHBOR', 'Desired_program_ELECTRONIC AUTOMATION TECHNOLOGY',
       'Desired_program_INFORMATIC ENGINEERING', 'Desired_program_UNSPECIFIED',
       'Father_level_HIGH SCHOOL', 'Father_level_PRIMARY SCHOOL',
       'Father_level_TECHNICAL', 'Father_level_TECHNOLOGIST',
       'Father_level_UNDERGRADUATE', 'Father_level_UNREGISTERED',
       'Mother_level_HIGH SCHOOL', 'Mother_level_PRIMARY SCHOOL',
       'Mother_level_TECHNICAL', 'Mother_level_TECHNOLOGIST',
       'Mother_level_UNDERGRADUATE', 'Mother_level_UNREGISTERED']
df = df[features]

# st.subheader('User Input features')
# st.write(df)
st.subheader("Prediction.")
load_clf = pickle.load(open('dropout_used_all.pkl', 'rb'))

prediction = load_clf.predict(df)
if prediction == 0:
    prediction = "Not Dropout"
else:
    prediction = "He will Dropout"
st.info(prediction)
