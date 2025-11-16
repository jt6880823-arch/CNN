import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10

# Load test data
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.title("CIFAR-10 CNN Classifier")

# Load trained model
model = load_model("cifar10_cnn.h5")  # Make sure this file exists

# Show random test image and prediction
if st.button("Show Random Test Image"):
    idx = np.random.randint(0, len(x_test))
    img = x_test[idx]
    st.image(img, use_column_width=True)
    
    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Predicted Label: {predicted_class}")
