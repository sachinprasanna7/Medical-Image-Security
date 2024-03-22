
# Code contributed by Abhayjit Singh Gulati (211IT085)

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

st.markdown("### Arnold Cat Map")

st.write("---")


st.subheader("Select the number of iterations")
iterations = st.slider("Select the number of iterations", 1, 192, 1)

st.write("---")

st.subheader("Upload an Image")
image = st.file_uploader("Upload Image", type=["png"])

st.write("---")

if image:
    st.subheader("Uploaded Image")

    col1, col2, col3 = st.columns(3)
    with col1:
     st.write(' ')
    with col2:         
         st.image(image, width=300)
    with col3:
     st.write(' ')

    save_path = "assets/input_image_arnold.png"  
    with open(save_path, "wb") as f:
        f.write(image.read())
else:
    st.subheader("Uploaded Image")
    st.warning("Please upload an image using the uploader above.")

st.write("---")
st.subheader("Arnold Scrambled Image")

if not image:
    st.warning("Please upload an image using the uploader above.")
else:
    with st.spinner("Performing Arnold Scrambling on the image. Please wait..."):
        os.system("jupyter nbconvert --to script modules/arnold_cat_map.ipynb")
        os.system(f"python modules/arnold_cat_map.py --iterations {iterations}")

    if os.path.exists("generated_assets/arnold.png"):      
      col1, col2, col3 = st.columns(3)
      with col1:
         st.write(' ')
      with col2:         
          st.image("generated_assets/arnold.png", use_column_width=True)
      with col3:
         st.write(' ')
    else:
        st.error("An error occurred while processing the image. Please try again.")





