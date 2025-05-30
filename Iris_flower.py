MIT License
Copyright (c) 2025 Aly Mostafa
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load and cache the Iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Title Screen
st.title(" 🌸Iris Flower Species Predictor🌸")
st.markdown(
    """
    Welcome to the **Iris Flower Species Predictor**!  
    This app uses a **Random Forest Classifier** to predict the species of an Iris flower based on its sepal and petal dimensions.  
    Adjust the sliders on the sidebar to input dimensions and see the prediction update dynamically.  
    ---
    """
)

# Sidebar Inputs
st.sidebar.title("🔧 Input Features")
st.sidebar.markdown("Adjust the sliders to input the flower's dimensions:")
sepal_length = st.sidebar.slider('Sepal Length (cm)', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider('Sepal Width (cm)', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider('Petal Length (cm)', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider('Petal Width (cm)', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

# Predict species based on user input
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Dynamic Output Section
st.markdown("### 🌟 Prediction Results")
st.write(f"The predicted species is: **{predicted_species}**")

# Highlight prediction dynamically
if predicted_species == "setosa":
    st.success("This flower is predicted to be **Setosa**! 🌺")
elif predicted_species == "versicolor":
    st.info("This flower is predicted to be **Versicolor**! 🌸")
else:
    st.warning("This flower is predicted to be **Virginica**! 🌼")

