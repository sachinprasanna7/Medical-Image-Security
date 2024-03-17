# basic streamlit configuration
import streamlit as st
import pandas as pd
import os
import nbformat
from nbconvert import ScriptExporter

# Set page configuration
st.set_page_config(
    page_title="Medical Image Security",
    page_icon=":hospital:",
    layout="wide",  # Use wide layout for better UI
)

st.title("Medical Image Security App")

st.markdown("### Secure and Manage Medical Images with Ease")

st.write("---")

# Upload image
image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

st.write("---")

# Define a section to display the uploaded image
if image:
    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    # Save the image to a specified location
    save_path = "assets/main_image.png"  # Specify your desired file path
    with open(save_path, "wb") as f:
        f.write(image.read())
else:
    st.subheader("Uploaded Image")
    st.markdown(
        "<div style='background-color: #ffe066; padding: 10px; border-radius: 5px; line-height: 5em; text-align: center; color: #333;'>Please upload an image using the uploader above.</div>",
        unsafe_allow_html=True
    )


st.write("---")

# Upload image
watermark_image = st.file_uploader("Upload Watermark Image", type=["jpg", "jpeg", "png"])

# Define a section to display the uploaded image and save it
if watermark_image:
    st.subheader("Uploaded Image")
    st.image(watermark_image, use_column_width=True)

    # Save the image to a specified location
    save_path_2 = "assets/watermark_image.png"  
    with open(save_path_2, "wb") as f:
        f.write(watermark_image.read())
else:
    st.subheader("Uploaded Image")
    st.markdown(
        "<div style='background-color: #ffe066; padding: 10px; border-radius: 5px; line-height: 5em; text-align: center; color: #333;'>Please upload an image using the uploader above.</div>",
        unsafe_allow_html=True
    )

st.write("---")
st.subheader("Encryption")

# run only if the image and watermark image are uploaded
if not image or not watermark_image:
    st.warning("Please upload an image using the uploader above.")
else:
    with st.spinner("Performing encryption on the image. Please wait..."):
        os.system("jupyter nbconvert --to script modules/endsem_notebook.ipynb")
        os.system("python modules/endsem_notebook.py")

    # if os.path.exists("generated_assets/rsvd_comparision.png"):   
    #     st.image("generated_assets/rsvd_comparision.png", use_column_width=True)
    # else:
    #     st.error("An error occurred while processing the image. Please try again.")


