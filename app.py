
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import pickle

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header{
font-size:3rem;
color:#6a0dad;
text-align:center;
margin-bottom:1rem;
}
.prediction-card{
background:#f3f0ff;
padding:20px;
border-radius:12px;
border-left:8px solid #6a0dad;
}
.footer{
text-align:center;
color:gray;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(fmt):
    if fmt=="joblib":
        return joblib.load("models/iris_model.joblib")
    with open("models/iris_model.pickle","rb") as f:
        return pickle.load(f)

@st.cache_data
def load_info():
    with open("models/model_info.json") as f:
        return json.load(f)

@st.cache_data
def load_ranges():
    with open("models/feature_ranges.json") as f:
        return json.load(f)

info=load_info()
ranges=load_ranges()

with st.sidebar:
    st.title("⚙️ Settings")
    fmt=st.radio("Model Format",["joblib","pickle"])
    model=load_model(fmt)
    st.divider()
    st.subheader("📊 Model Information")
    st.write(f"**Model:** {info['model_type']}")
    st.write(f"**Accuracy:** {info['accuracy']:.2%}")
    st.write(f"**Features:** {len(info['feature_names'])}")
    st.write(f"**Classes:** {len(info['target_names'])}")

st.markdown('<h1 class="main-header">🌸 Iris Flower Classification</h1>',unsafe_allow_html=True)
st.write("Predict the species of an Iris flower using a trained Random Forest model.")

left,right=st.columns([2,1])

with left:
    st.subheader("📝 Input Features")
    sl=st.slider("Sepal Length (cm)",float(ranges["sepal_length"]["min"]),float(ranges["sepal_length"]["max"]),float(ranges["sepal_length"]["default"]),0.1)
    sw=st.slider("Sepal Width (cm)",float(ranges["sepal_width"]["min"]),float(ranges["sepal_width"]["max"]),float(ranges["sepal_width"]["default"]),0.1)
    pl=st.slider("Petal Length (cm)",float(ranges["petal_length"]["min"]),float(ranges["petal_length"]["max"]),float(ranges["petal_length"]["default"]),0.1)
    pw=st.slider("Petal Width (cm)",float(ranges["petal_width"]["min"]),float(ranges["petal_width"]["max"]),float(ranges["petal_width"]["default"]),0.1)

with right:
    st.subheader("📋 Current Values")
    st.dataframe(pd.DataFrame({
        "Feature":["Sepal Length","Sepal Width","Petal Length","Petal Width"],
        "Value":[sl,sw,pl,pw]
    }),hide_index=True,use_container_width=True)

sample=np.array([[sl,sw,pl,pw]])

if st.button("🎯 Predict Species",type="primary",use_container_width=True):
    pred=model.predict(sample)[0]
    prob=model.predict_proba(sample)[0]
    species=info["target_names"][pred]
    st.markdown('<div class="prediction-card">',unsafe_allow_html=True)
    st.subheader("Prediction")
    st.success(f"Predicted Species: **{species.capitalize()}**")
    st.subheader("📈 Confidence Scores")
    for i,name in enumerate(info["target_names"]):
        st.write(name.capitalize())
        st.progress(float(prob[i]))
        st.write(f"{prob[i]*100:.2f}%")
    st.markdown("</div>",unsafe_allow_html=True)

with st.expander("📚 About Iris Dataset"):
    st.markdown("""
- **150 samples**
- **4 numerical features**
- **3 flower species**
- **Random Forest Classifier**
- Species:
  - Setosa
  - Versicolor
  - Virginica
""")

st.markdown("---")
st.markdown("<div class='footer'>Built with Streamlit & Scikit-learn</div>",unsafe_allow_html=True)