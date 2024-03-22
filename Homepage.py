
# Code contributed by Sachin Prasanna (211IT058), Abhayjit Singh Gulati (211IT085)

import streamlit as st
import pandas as pd
import os
import nbformat
from nbconvert import ScriptExporter
import cv2

# Setting page configuration
st.set_page_config(
    page_title="Medical Image Security",
    page_icon=":hospital:",
    layout="wide",  
)
css = """
<style>
    /* Center align the app title */
    .title {
        text-align: center;
        color: black;
        margin-bottom: 20px;
        font-size: 32px;
    }
    /* Styling for the department info */
    .info {
    color: black;
    text-align: center;
    margin-bottom: 10px;
    display: inline-block;
    vertical-align: middle;
    margin-left: 15%; /* Add margin to create spacing */
    font-size: 24px;
    # font-family: 'Times New Roman', Times, serif; /* Set font style to Times New Roman */
}
    /* Logo and text container */
    .logo-container {
        text-align: left;
    }
    .logo-container img {
        vertical-align: middle;
    }
    /* Footer styling */
    .footer p {
        margin: 5px 0;
    }
    .footer p:hover {
        color: #ddd; /* Change text color on hover */
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.markdown("""
<div class="logo-container">
    <img src="https://upload.wikimedia.org/wikipedia/en/c/cc/NITK_Emblem.png" width="100" class="logo" />
    <div class="info">
        <div>National Institute of Technology Karnataka</div>
        <div>Department of Information Technology</div>
        <div>Information Assurance and Security</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("---")
st.markdown("<h1 class='title'>Medical Image Security App</h1>", unsafe_allow_html=True)

footer_html = """
    <div style="background-color:#82CEC4;padding:20px;border-radius:10px;color:black; text-align:center;">
        <p style="font-size:25px;"> <b>Guide</b>: Dr. Jaidhar C.D.</p>
        <p style="font-size:23px;"> <b>Implemented by:</b> Sachin Prasanna (211IT058), Rounak Jain (211IT055), Abhayjit Singh Gulati (211IT085)</p>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)



st.write("---")

st.subheader("Upload Image")
image = st.file_uploader("", type=["png"])

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
   
    save_path = "assets/main_image.png" 
    with open(save_path, "wb") as f:
        f.write(image.read())
else:
    st.subheader("Uploaded Image")
    st.warning("Please upload an image using the uploader above.")


st.write("---")

st.subheader("Upload Watermark Image")
watermark_image = st.file_uploader("Watermark", type=["png"])

st.write("---")

if watermark_image:
    st.subheader("Watermark Image")
    col1, col2, col3 = st.columns(3)
    with col1:
     st.write(' ')
    with col2:         
         st.image(watermark_image, width=300)
    with col3:
     st.write(' ')
 
    save_path_2 = "assets/watermark_image.png"  
    with open(save_path_2, "wb") as f:
        f.write(watermark_image.read())
else:
    st.subheader("Uploaded Image")
    st.warning("Please upload an image using the uploader above.")

st.write("---")
st.subheader("Encryption")

# run only if the image and watermark image are uploaded
if not image or not watermark_image:
    st.warning("Please upload an image using the uploader above.")
else:
    with st.spinner("Performing encryption on the image. Please wait..."):
        os.system("jupyter nbconvert --to script modules/endsem_notebook.ipynb")
        os.system("python modules/endsem_notebook.py")
    
    if os.path.exists("generated_assets/final_encrypted_image.png"):
        st.subheader("Encrypted Image")
        col1, col2, col3 = st.columns(3)
        with col1:
           st.write(' ')
        with col2:         
          st.image("generated_assets/final_encrypted_image.png", use_column_width=True)
        with col3:
          st.write(' ')
        
        st.write("---")
        st.subheader("Decryption")
        if os.path.exists("generated_assets/final_recovered_watermark_image.png"):
                    st.subheader("Decrypted Image")
                    st.image("generated_assets/final_recovered_watermark_image.png", use_column_width=True)
        else:
                    st.warning("The decrypted image is not available.")


        
# Footer
st.markdown("---")
footer_html = """
    <div style="background-color:#82CEC4;padding:20px;border-radius:10px;color:black;">
        <h6 style="color:black;"> Paper Implemented:</h4>
        <p>Singh, Om Prakash, et al. "SecDH: Security of COVID-19 images based on data hiding with PCA." Computer Communications 191 (2022): 368-377.</p>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)