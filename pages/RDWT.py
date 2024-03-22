
# Code contributed by Abhayjit Singh Gulati (211IT055)

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

st.markdown("### RDWT")

st.write("---")

image = st.file_uploader("Upload Image", type=["png"])

st.write("---")

if image:
    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)
    save_path = "assets/input_image_rdwt.png"  
    with open(save_path, "wb") as f:
        f.write(image.read())
else:
    st.subheader("Uploaded Image")
    st.warning("Please upload an image using the uploader above.")

st.write("---")
st.subheader("RDWT Image")

if not image:
    st.warning("Please upload an image using the uploader above.")
else:
    with st.spinner("Performing RDWT on the image. Please wait..."):
        os.system("jupyter nbconvert --to script modules/rdwt.ipynb")
        os.system("python modules/rdwt.py")

    if os.path.exists("generated_assets/image_rdwt.png"):    
        st.image("generated_assets/image_rdwt.png", use_column_width=True)
        st.write("---")
        st.subheader("RDWT Image (Inverse)")
        if os.path.exists("generated_assets/irdwt.png"):      
              st.image("generated_assets/irdwt.png", use_column_width=True)
    else:
        st.error("An error occurred while processing the image. Please try again.")







