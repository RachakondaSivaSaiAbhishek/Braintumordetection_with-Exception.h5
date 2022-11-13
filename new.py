import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('tumor.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Brain Tumor Detection
         """
         )

file = st.file_uploader("Upload the image to be classified \U0001F447", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def upload_predict(upload_image, model):
  size = (128,128)    
  image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
  image = np.asarray(image)
  img_reshape = image.reshape(1,128,128,3)

  prediction = model.predict(img_reshape)
  
  return prediction
if file is None:
  st.text("Please upload an image file")
else:
  img = Image.open(file)
  st.image(img, use_column_width=True)
  predictions = upload_predict(img, model)
  predict=np.argmax(predictions[0])
  if predict==1:
      st.write("Tumor Detected")
  else:
      st.write("No tumor detected")