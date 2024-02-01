import streamlit as st
import numpy as np
from PIL import Image


import tensorflow as tf


from tensorflow import keras

# Load the saved model
model = keras.models.load_model("Tumor_classifier_model.h5")

# Define a function to make predictions on new images
def predict(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Scale pixel values to [0, 1]
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    prediction = model.predict(img_array)
    return prediction[0][0]

# Use Streamlit to create a file uploader and make predictions on uploaded images
st.title("Brain Tumor Classifier")
uploaded_file = st.file_uploader("Choose the MRL Image Only.....", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save the uploaded file to disk
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Make a prediction on the uploaded image
    prediction = predict(image_path)
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    if prediction > 0.5:
        st.write("The model predicts that this image contains a brain tumor with {:.2f}% confidence.".format(prediction*100))
    else:
        st.write("The model predicts that this image does not contain a brain tumor with {:.2f}% confidence.".format((1-prediction)*100))
