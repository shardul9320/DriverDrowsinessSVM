import streamlit as st
import pickle
import cv2
import numpy as np
from skimage.feature import hog
from PIL import Image

# Load the SVM model and Label Encoder from local files
@st.cache_resource
def load_local_models():
    try:
        # Provide relative or absolute paths
        model_path = "svm_model.pkl"
        encoder_path = "label_encoder.pkl"
        
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

        with open(encoder_path, "rb") as encoder_file:
            label_encoder = pickle.load(encoder_file)

        return model, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Extract HOG features
def extract_hog_features(image):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (64, 128))
        fd = hog(gray_image, block_norm="L2-Hys", visualize=False)
        return fd
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return None

# Predict the class based on the uploaded image
def predict_image(image, model, label_encoder):
    try:
        # Convert the PIL Image to a NumPy array
        image_np = np.array(image)
        # Extract features
        feature_vector = extract_hog_features(image_np)
        if feature_vector is None:
            return "Error in feature extraction."
        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)
        # Make a prediction
        prediction = model.predict(feature_vector)
        predicted_label = label_encoder.inverse_transform(prediction)
        return predicted_label[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error during prediction."

# Streamlit UI layout
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide", page_icon="🚗")

# Sidebar
st.sidebar.title("Instructions 🚦")
st.sidebar.write(
    """
    1. Upload an image of a driver.
    2. The app will analyze the image to detect yawning.
    3. Supported formats: JPG, JPEG, PNG.
    """
)

st.sidebar.info("This app uses HOG features and an SVM classifier for detection.")

# Header
st.title("🚗 Driver Drowsiness Detection")
st.subheader("Predict yawning using an SVM model and HOG features.")
st.write(
    """
    Fatigue is a leading cause of road accidents. This app uses machine learning techniques 
    to help identify if a driver is yawning based on an uploaded image. Stay safe on the road!
    """
)

# Load models
model, label_encoder = load_local_models()

if model is None or label_encoder is None:
    st.error("Failed to load the SVM model or label encoder. Please ensure they are correctly uploaded.")
else:
    # Main area for file upload
    uploaded_file = st.file_uploader("Upload a driver's image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict the class of the uploaded image
        with st.spinner("Analyzing the image..."):
            predicted_label = predict_image(image, model, label_encoder)

        # Display the prediction
        if predicted_label:
            st.success(f"Prediction: **{predicted_label}**")
            if predicted_label.lower() == "yawn":
                st.warning("Warning: The driver appears to be yawning. Take appropriate action!")
            else:
                st.success("No signs of yawning detected. Drive safely!")
    else:
        st.info("Please upload an image to get started.")









# import streamlit as st
# import pickle
# import cv2
# import numpy as np
# from skimage.feature import hog
# from PIL import Image

# # Load the SVM model and Label Encoder from local files
# @st.cache_resource
# def load_local_models():
#     with open("svm_model.pkl", "rb") as model_file:
#         model = pickle.load(model_file)

#     with open("label_encoder.pkl", "rb") as encoder_file:
#         label_encoder = pickle.load(encoder_file)

#     return model, label_encoder

# # Extract HOG features
# def extract_hog_features(image):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Resize the image to match the expected input size for HOG
#     gray_image = cv2.resize(gray_image, (64, 128))
#     # Extract HOG features
#     fd, _ = hog(gray_image, block_norm="L2-Hys", visualize=True)
#     return fd

# # Function to predict the class based on the uploaded image
# def predict_image(image, model, label_encoder):
#     # Convert the PIL Image to a NumPy array
#     image_np = np.array(image)
#     # Extract features
#     feature_vector = extract_hog_features(image_np)
#     # Reshape for prediction
#     feature_vector = feature_vector.reshape(1, -1)

#     # Make a prediction
#     prediction = model






# import streamlit as st
# import pickle
# import requests
# import cv2
# import numpy as np
# from skimage.feature import hog
# from PIL import Image
# import label_encoder.pkl as label_encoder
# import svm_model.pkl as svm_model

# # Function to load the model from GitHub
# def load_model_from_github(model_url, encoder_url):
#     model_response = requests.get(model_url)
#     encoder_response = requests.get(encoder_url)

#     with open('svm_model.pkl', 'wb') as f:
#         f.write(model_response.content)
#     with open('label_encoder.pkl', 'wb') as f:
#         f.write(encoder_response.content)

#     with open('svm_model.pkl', 'rb') as model_file:
#         model = pickle.load(model_file)

#     with open('label_encoder.pkl', 'rb') as encoder_file:
#         label_encoder = pickle.load(encoder_file)

#     return model, label_encoder

# # Extract HOG features
# def extract_hog_features(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray_image = cv2.resize(gray_image, (64, 128))
#     fd, _ = hog(gray_image, block_norm='L2-Hys', visualize=True)
#     return fd

# # Function to predict the class based on the uploaded image
# def predict_image(image_path, model, label_encoder):
#     image = cv2.imread(image_path)
#     feature_vector = extract_hog_features(image)
#     feature_vector = feature_vector.reshape(1, -1)

#     prediction = model.predict(feature_vector)
#     predicted_label = label_encoder.inverse_transform(prediction)

#     return predicted_label[0]

# # Streamlit app layout
# st.title("Yawn Detection")

# # GitHub URLs for the model and label encoder files
# # model_url = 'https://github.com/your-username/your-repository/raw/main/svm_model.pkl'
# # encoder_url = 'https://github.com/your-username/your-repository/raw/main/label_encoder.pkl'

# # Load model and label encoder
# # model, label_encoder = load_model_from_github(svm_model,label_encoder)

# # File uploader to upload image
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert the uploaded file to an image
#     image = Image.open(uploaded_file)
#     image_path = "uploaded_image.jpg"
#     image.save(image_path)

#     # Predict the class of the uploaded image
#     predicted_label = predict_image(image_path, svm_model, label_encoder)

#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     st.write(f"Prediction: {predicted_label}")
