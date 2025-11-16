import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Load CIFAR-10 dataset
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
y_test = to_categorical(y_test, 10)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Build the same model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

st.title("CIFAR-10 CNN Classifier")

st.write("This is a simple CNN trained on CIFAR-10 dataset.")

# Option to select a random test image
if st.button("Show Random Test Image"):
    idx = np.random.randint(0, len(x_test))
    img = x_test[idx]
    st.image(img, caption=f"True Label: {class_names[np.argmax(y_test[idx])]}", use_column_width=True)
    
    # Predict
    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Predicted Label: {predicted_class}")
