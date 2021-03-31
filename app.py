#import necessay libraries
import streamlit as st
import numpy as np
#import pandas as pd
from PIL import Image
import tensorflow as tf
#import cv2
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
#from numpy import asarray
from skimage.transform import resize

#this set option is for ignoring warning
st.set_option("deprecation.showfileUploaderEncoding", False)


#this cache option will make the model to run once and save it.otherwise it will take a long time if model runs everytime.
@st.cache(allow_output_mutation=True)
def load_model():
    #this hdf5 file was created from the compiled model that we built in jupyter notebook
    model=tf.keras.models.load_model('./model_x.hdf5')
    return model
model=load_model()


#this function returns the class. there are two classes. mask and without mask. this function takes loaded image that has been covereted to numpy array as parameter
def test_mask(path):
    #this line adds extra dimension to the numpy array of corresponding image
    y = np.expand_dims(path, axis=0)
    images1 = np.vstack([y])
    classes = model.predict_classes(images1, batch_size=10)
    return (classes[0][0])


#main title of the app and description   
st.title("Detect mask from image/video/webcam")
st.write("Choose one of the following options to proceed:") 
#three radio buttons with three imput options. 
choice = st.radio("", ("Choose to upload an image","Choose a video","Choose your webcam"))


#if you choose one radio button for uploading image then this event happens
if choice == "Choose to upload an image":
    image = st.file_uploader("Upload", type=['jpg','png','jpeg'])
    
    #if image is uploaded then it opens the image and show it
    if image is not None:
        u_img = Image.open(image)
        show = st.image(image, use_column_width=True)
        show.image(u_img, 'Uploaded Image', use_column_width=True)
        
        # We preprocess the image to fit in algorithm.image is convertred to corresponding numpy array
        image = np.asarray(u_img)
        
        #numpy array of the image needs to resize according to our train input size
        image1= resize(image, (64,64))
        
        #st.write("shape of input image : ",image1.shape)
        string=test_mask(image1)
        if string==0:
            st.success("Mask detected")
        else:
            st.success("No mask detected")