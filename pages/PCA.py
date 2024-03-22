
# Code contributed by Rounak Jain (211IT055)

import streamlit as st
import pandas as pd
import os
import nbformat
from nbconvert import ScriptExporter

st.set_page_config(
    page_title="Medical Image Security",
    page_icon=":hospital:",
    layout="wide",  
)

st.title("Medical Image Security App")

st.markdown("### Principal Component Analysis (PCA)")

st.write("---")

image = st.file_uploader("Upload Image", type=["png"])

st.write("---")

if image:
    st.subheader("Uploaded Image")
    col1, col2, col3 = st.columns(3)
    with col1:
          st.write(' ')
    with col2:         
           st.image(image, use_column_width=True)
    with col3:
          st.write(' ')

    save_path = "assets/input_image_pca.png"  
    with open(save_path, "wb") as f:
        f.write(image.read())
else:
    st.subheader("Uploaded Image")
    st.warning("Please upload an image using the uploader above.")

st.write("---")
st.subheader("PCA")

if not image:
    st.warning("Please upload an image using the uploader above.")
else:
    with st.spinner("Performing PCA. Please wait..."):
        os.system("jupyter nbconvert --to script modules/pca.ipynb")
        os.system("python modules/pca.py")

    if os.path.exists("generated_assets/output_image_pca.png"):    
        
        col1, col2, col3 = st.columns(3)
        with col1:
          st.write(' ')
        with col2:     
            st.image("generated_assets/output_image_pca.png", use_column_width=True)
        with col3:
          st.write(' ')
    else:
         st.error("An error occurred while processing the image. Please try again.")







