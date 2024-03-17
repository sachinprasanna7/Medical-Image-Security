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

st.markdown("### Arnold Cat Map")

st.write("---")


st.subheader("Select the number of iterations")
iterations = st.slider("Select the number of iterations", 1, 192, 1)

st.write("---")

# Upload image
st.subheader("Upload an Image")
image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

st.write("---")

# Define a section to display the uploaded image
if image:
    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    # Save the image to a specified location
    save_path = "assets/input_image_arnold.png"  # Specify your desired file path
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
      st.image("generated_assets/arnold.png", use_column_width=True)

    #   st.write("---")
    #   st.subheader("Arnold Scrambled Image (Inverse)")
    #   if st.button("Perform Inverse Arnold Scrambling"):
    #         if os.path.exists("generated_assets/inverse_arnold.png"):      
    #             st.image("generated_assets/inverse_arnold.png", use_column_width=True)
    else:
        st.error("An error occurred while processing the image. Please try again.")





