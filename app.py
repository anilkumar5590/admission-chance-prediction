import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(
        page_title="Admission Chance Prediction",
)
# Load the dataset
chance = pd.read_csv('Admission_Chance.csv')

st.title('Admission Chance Prediciton')
# Define the input fields using Streamlit's number_input
gre = st.number_input('Enter Graduate Record Examination (GRE) Score')
toefl = st.number_input('Enter Test of English as a Foreign Language (TOEFL) Score')
ur = st.number_input('Enter University Rating')
sop = st.number_input('Enter Statement of Purpose (SOP)')
lor = st.number_input('Enter Letter of Recommendation Strength (LOR)')
cgpa = st.number_input('Enter Undergraduate CGPA')
research = st.number_input('Enter Research Experience')

# Separate the features (X) and target variable (y)
y = chance['Chance of Admit ']
X=chance[['GRE Score', 'TOEFL Score', 'University Rating', ' SOP',
       'LOR ', 'CGPA', 'Research']]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2529)

# Create a LinearRegression model instance and fit it to the training data
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Collect user input into a list
user_input = [gre, toefl, ur, sop, lor, cgpa, research]

# Make prediction when the button is clicked
if st.button('Make Prediction'):
    prediction = linear_regression.predict([user_input]) * 100
    st.write(f"The Admission Chance is  {prediction[0]:.2f}%")
