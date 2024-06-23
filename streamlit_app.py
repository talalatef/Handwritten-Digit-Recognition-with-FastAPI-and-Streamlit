import streamlit as st
import requests
from PIL import Image
import numpy as np

# URL of the FastAPI server
url = "http://localhost:8000/predict/"

def predict_image(image):
    # Convert the image to binary format
    with st.spinner("Making prediction..."):
        image_file = image.convert("L").resize((28, 28))
        image_file = np.array(image_file, dtype=np.uint8)
        image_file = Image.fromarray(image_file)

        with open("temp_image.png", "wb") as f:
            image_file.save(f, format="PNG")

        with open("temp_image.png", "rb") as f:
            response = requests.post(url, files={"file": f})
            prediction = response.json()
        
        return prediction

# CSS for background color
page_bg_color = '''
<style>
body {
    background-color: #8B0000;  /* Dark Red Color */
}
</style>
'''

st.markdown(page_bg_color, unsafe_allow_html=True)

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = predict_image(image)
    st.write(f"Prediction: {prediction['prediction']}")
