# basic streamlit configuration
import streamlit as st
import pandas as pd
import os
import nbformat
from nbconvert import ScriptExporter
import cv2

# Set page configuration
st.set_page_config(
    page_title="Medical Image Security",
    page_icon=":hospital:",
    layout="wide",  # Use wide layout for better UI
)

# Define CSS styles
css = """
    <style>
        /* Center align the app title */
        .title {
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }
        /* Styling for the container */
        .container {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        /* Styling for the department info */
        .info {
            color: white;
            text-align: center;
            margin-bottom: 10px;
        }
        /* Logo styling */
        .logo {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .footer p {
                margin: 5px 0;
            }
            .footer p:hover {
                color: #ddd; /* Change text color on hover */
            }
    </style>
"""

# Render CSS
st.markdown(css, unsafe_allow_html=True)

# App title and information
st.markdown("<h1 class='title'>Medical Image Security App</h1>", unsafe_allow_html=True)
st.write("---")
st.markdown("<div class='info'>National Institute of Technology Karnataka</div>", unsafe_allow_html=True)
st.markdown("<div class='info'>Department of Information Technology</div>", unsafe_allow_html=True)
st.markdown("<div class='info'>Information Assurance and Security</div>", unsafe_allow_html=True)

# st.markdown("<div class='info'>National Institute of Technology, Karnataka</div>", unsafe_allow_html=True)


# Image logo
col1, col2, col3 = st.columns([1.5, 1, 1])
with col2:
    st.image("nitk_logo.png", width=100)


# Separator
st.write("---")

# Upload image
image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

st.write("---")

# Define a section to display the uploaded image
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

# Upload image
watermark_image = st.file_uploader("Upload Watermark Image", type=["jpg", "jpeg", "png"])

st.write("---")

# Define a section to display the uploaded image and save it
if watermark_image:
    st.subheader("Watermark Image")
    col1, col2, col3 = st.columns(3)
    with col1:
     st.write(' ')
    with col2:         
         st.image(watermark_image, width=300)
    with col3:
     st.write(' ')
 

    # st.image(watermark_image, use_column_width=True)

    # Save the image to a specified location
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
        st.image("generated_assets/final_encrypted_image.png", use_column_width=True)
        
        st.write("---")
        st.subheader("Final Results")
        if os.path.exists("generated_assets/final_result.png"):
            st.image("generated_assets/final_result.png", use_column_width=True)
        else:
            st.warning("The final result image is not available.")
        
        st.write("---")
        st.subheader("Decryption")
            # button to decrypt the image
        if os.path.exists("generated_assets/final_recovered_watermark_image.png"):
                    st.subheader("Decrypted Image")
                    st.image("generated_assets/final_recovered_watermark_image.png", use_column_width=True)
        else:
                    st.warning("The decrypted image is not available.")


        
# Footer
st.markdown("---")


footer_html = """
    <div style="background-color:#0c0814;padding:20px;border-radius:10px;color:white;">
        <h3>Developed By:</h3>
        <p>Sachin Prasanna</p>
        <p>Rounak Jain</p>
        <p>Abhayjit Singh Gulati</p>
        <h3>Under the guidance of:</h3>
        <p>Dr. Jaidhar C.D.</p>
        <h4> Paper Implemented:</h4>
        <p>Singh, Om Prakash, et al. "SecDH: Security of COVID-19 images based on data hiding with PCA." Computer Communications 191 (2022): 368-377.</p>
    </div>
"""

st.markdown(footer_html, unsafe_allow_html=True)