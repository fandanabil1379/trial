import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Load the model
model = pickle.load(open('model/final_model.sav', 'rb'))

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def main():
    # Load picture
    image_hospital = Image.open('img/hospital.jpg')

    # Add option to select online or offline prediction
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch")
        )

    # Add explanatory text and picture in the sidebar
    st.sidebar.info('This app is created to predict diabetes patient')    
    st.sidebar.image(image_hospital)

    # Add title
    st.title("Diabetes Prediction App")

    if add_selectbox == 'Online':

        # Set up the form to fill in the required data 
        pregnancies = st.number_input(
            'Pregnancies', min_value=0, max_value=17)
        glucose = st.number_input(
            'Glucose', min_value=0, max_value=200)
        bloodPressure = st.number_input(
            'Blood Pressure', min_value=0, max_value=125)
        skinThickness = st.number_input(
            'Skin Thickness', min_value=0, max_value=100)
        insulin = st.number_input(
            'Insulin', min_value=0, max_value=900)
        bmi = st.number_input(
            'BMI', min_value=0, max_value=70)
        diabetesPedigreeFunction = st.number_input(
            'Diabetes Pedigree Function', min_value=0, max_value=3)
        age = st.number_input(
            'Age', min_value=0, max_value=100)  
    
        # Convert form to data frame
        input_df = pd.DataFrame([
            {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': bloodPressure,
                'SkinThickness': skinThickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetesPedigreeFunction,
                'Age': age
                }
            ]
        )
        
        # Set a variabel to store the output
        output = ""

        # Make a prediction 
        if st.button("Predict"):
            output = model.predict(input_df)
            if (output[0] == 0):
                output = 'The person is not diabetic'
            else:
                output = 'The person is diabetic'

        # Show prediction result
        st.success(output)          

    if add_selectbox == 'Batch':

        # Add a feature to upload the file to be predicted
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            # Convert the file to data frame
            data = pd.read_csv(file_upload)

            # Select only columns required by the model
            data = data[[
                'Pregnancies',
                'Glucose',
                'BloodPressure',
                'SkinThickness',
                'Insulin',
                'BMI',
                'DiabetesPedigreeFunction',
                'Age'
                ]
            ]


            # Make predictions
            data['Prediction'] = np.where(model.predict(data)==1, 'Diabetic', 'Non Diabetic')

            # Show the result on page
            st.write(data)

            # Add a button to download the prediction result file 
            st.download_button(
                "Press to Download",
                convert_df(data),
                "Prediction Result.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == '__main__':
    main()