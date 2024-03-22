
# Code contributed by Sachin Prasanna (211IT058)

import streamlit as st
import pandas as pd
import os
import nbformat
from nbconvert import ScriptExporter

# Set page configuration
st.set_page_config(
    page_title="Medical Image Security",
    page_icon=":hospital:",
    layout="wide",  
)

st.title("Medical Image Security App")

st.markdown("### RSVD")

st.write("---")

image = st.file_uploader("Upload Image", type=["png"])

st.write("---")

if image:
    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)
    
    save_path = "assets/rsvd.png" 
    with open(save_path, "wb") as f:
        f.write(image.read())
else:
    st.subheader("Uploaded Image")
    st.warning("Please upload an image using the uploader above.")

st.write("---")
st.subheader("RSVD")

if not image:
    st.warning("Please upload an image using the uploader above.")
else:
    with st.spinner("Performing RSVD on the image. Please wait..."):
        os.system("jupyter nbconvert --to script modules/rsvd.ipynb")
        os.system("python modules/rsvd.py")

    if os.path.exists("generated_assets/rsvd_comparision.png"):      
        
        st.image("generated_assets/rsvd_comparision.png", use_column_width=True)
    else:
        st.error("An error occurred while processing the image. Please try again.")




