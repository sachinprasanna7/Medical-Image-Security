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
    save_path = "cameraman.jpg"  # Specify your desired file path
    with open(save_path, "wb") as f:
        f.write(image.read())
else:
    st.subheader("Uploaded Image")
    st.warning("Please upload an image using the uploader above.")

st.write("---")
# create a button to run the normalisation.ipynb
# if st.button("Normalisation"):
#     os.system("jupyter nbconvert --to script normalisation.ipynb")
#     os.system("python normalisation.py")


