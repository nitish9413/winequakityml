import streamlit as st
import pandas as pd
import numpy as np
# import logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

st.title('Wine Quality Prediction')

# Load the data
df = pd.read_csv('winequality-red.csv')

# Show the data as a table
st.subheader('Data Information')
st.dataframe(df)

# Show statistics on the data
st.subheader('Data Statistics')
st.write(df.describe())

# Show the data as a chart
# st.subheader('Data Chart')
# chart = st.bar_chart(df)

# Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:11].values
Y = df.iloc[:, 11].values

# Split the dataset into 75% Training set and 25% Testing set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert quality into 2 classes
df['quality'] = [1 if x>5 else 0 for x in df['quality']]

# Create a function with many Machine Learning Models
def get_models():
    
    models = {}
    models['Logistic Regression'] = LogisticRegression()
    models['Random Forest'] = RandomForestClassifier()
    models['SVM'] = SVC()
    return models

# Create a function to train the models 
def train_model(models, X_train, Y_train):
    # Train the models and store in a dictionary
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, Y_train)
        trained_models[name] = model
    return trained_models

# Create a function to get the accuracy of the models
def get_accuracy(models, X_train, Y_train, X_test, Y_test):
    # Get the accuracy score of each model and store in a dictionary
    accuracy = {}
    for name, model in models.items():
        accuracy[name] = model.score(X_test, Y_test)
    return accuracy

# Get the models
models = get_models()

# Train the models
trained_models = train_model(models, X_train, Y_train)

# Get the accuracy of the models
accuracy = get_accuracy(trained_models, X_train, Y_train, X_test, Y_test)

# Show the models as a dropdown
st.subheader('Select Classifier')
classifier = st.selectbox('Classifier', ('Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM'))

# Show the accuracy of the selected model
st.subheader('Model Accuracy')
st.write(str(accuracy[classifier]))



# Predict the quality of the wine
st.subheader('Predict the Quality of Wine')
fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=15.0, value=0.0)
volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.0)
citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=1.0, value=0.0)
residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=15.0, value=0.0)
chlorides = st.number_input('Chlorides', min_value=0.0, max_value=1.0, value=0.0)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, max_value=60.0, value=0.0)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, max_value=300.0, value=0.0)
density = st.number_input('Density', min_value=0.0, max_value=2.0, value=0.0)
pH = st.number_input('pH', min_value=0.0, max_value=5.0, value=0.0)
sulphates = st.number_input('Sulphates', min_value=0.0, max_value=2.0, value=0.0)
alcohol = st.number_input('Alcohol', min_value=0.0, max_value=15.0, value=0.0)

# Store the user input into a variable
user_input = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]

# Scale the user input
user_input = sc.transform(user_input)

# Show the predicted quality
st.subheader('Predicted Quality')
st.write(str(trained_models[classifier].predict(user_input)))

# Show the probability of each class
st.subheader('Prediction Probability')
st.write(str(trained_models[classifier].predict_proba(user_input)))

# Save the model

pickle.dump(trained_models[classifier], open('model.pkl', 'wb'))

# Load the model that you want to use
model = pickle.load(open('model.pkl', 'rb'))

# Predict the output
