import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the fine-tuned VGG16 model
model = tf.keras.models.load_model('my_model_vgg.h5')

# Function to preprocess and predict the image
def predict_image(img):
    # Preprocess the image
    img = image.load_img(img, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescaling
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get the predicted class label
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return predicted_class, prediction

# Set page title
st.title("Image Classification with VGG16")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Perform prediction
    predicted_class, prediction = predict_image(uploaded_file)
    
    # Display prediction results
    st.write(f"Predicted class index: {predicted_class}")
    # Replace 'class_names' with the actual list of class names used during training
    class_names = ['Potosiabre vitarsis', 'Thrips', 'Xylotrechus', 'army worm']  # Replace with actual class names
    st.write(f"Predicted class : {class_names[predicted_class]}")
    st.write(f"Prediction probabilities: {prediction}")
