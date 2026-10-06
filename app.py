
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🌱",
    layout="centered"
)

model = joblib.load("dry_bean_svm_model.pkl")
scaler = joblib.load("dry_bean_scaler.pkl")

st.title("🌱 Dry Bean Type Classification")

st.write(
    "Enter the physical and geometrical measurements of a dry bean "
    "to predict its class."
)

st.subheader("Enter Bean Measurements")

Area = st.number_input("Area", value=44652.0)
Perimeter = st.number_input("Perimeter", value=794.941)
MajorAxisLength = st.number_input("Major Axis Length", value=296.883367)
MinorAxisLength = st.number_input("Minor Axis Length", value=192.431733)
AspectRation = st.number_input("Aspect Ratio", value=1.551124)
Eccentricity = st.number_input("Eccentricity", value=0.764441)
ConvexArea = st.number_input("Convex Area", value=45178.0)
EquivDiameter = st.number_input("Equivalent Diameter", value=238.438026)
Extent = st.number_input("Extent", value=0.759859)
Solidity = st.number_input("Solidity", value=0.988283)
roundness = st.number_input("Roundness", value=0.883157)
Compactness = st.number_input("Compactness", value=0.801277)
ShapeFactor1 = st.number_input("Shape Factor 1", value=0.006645)
ShapeFactor2 = st.number_input("Shape Factor 2", value=0.001694)
ShapeFactor3 = st.number_input("Shape Factor 3", value=0.642044)
ShapeFactor4 = st.number_input("Shape Factor 4", value=0.996386)

if st.button("Predict Bean Class"):

    input_data = pd.DataFrame([[
        Area,
        Perimeter,
        MajorAxisLength,
        MinorAxisLength,
        AspectRation,
        Eccentricity,
        ConvexArea,
        EquivDiameter,
        Extent,
        Solidity,
        roundness,
        Compactness,
        ShapeFactor1,
        ShapeFactor2,
        ShapeFactor3,
        ShapeFactor4
    ]], columns=[
        'Area',
        'Perimeter',
        'MajorAxisLength',
        'MinorAxisLength',
        'AspectRation',
        'Eccentricity',
        'ConvexArea',
        'EquivDiameter',
        'Extent',
        'Solidity',
        'roundness',
        'Compactness',
        'ShapeFactor1',
        'ShapeFactor2',
        'ShapeFactor3',
        'ShapeFactor4'
    ])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    st.success(f"Predicted Bean Class: {prediction}")
