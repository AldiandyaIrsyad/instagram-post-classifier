import pickle
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import PIL.Image

class columnDropperTransformer():
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self

model = tf.keras.models.load_model('model.h5')




def predict(img):
    def preproces_image(img):
        width, height = img.size
        if width > height:
            left = (width - height) // 2
            right = width - left
            img = img.crop((left, 0, right, height))
        elif height > width:
            total_cropped = height - width 
            top = total_cropped * (30/100)
            bottom = top + width

            img = img.crop((0, top, width, bottom))
        
        # resize to 224x224
        return img.resize((224, 224), PIL.Image.ANTIALIAS)
    # preprocessor

    # input --> model training
    # SAMA PERSIS
    img = preproces_image(img)
    img = np.expand_dims(img, axis=0)
    y_pred = model.predict(img).argmax(axis=1)[0]
    return y_pred


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Which Justin Posted That Picture?</h1> 
    </div> 
    """


    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    upload = st.file_uploader("Upload an img", type=["jpg", "png"])


    # # rewrite to
    if st.button("Predict"):
        # pass
        image = PIL.Image.open(upload)
        classname = ('justinbieber', 'justintimberlake', 'justinhartley', 'justinpjtrudeau', 'justinlong')
        pred = predict(image)
        st.write("")
        st.write("Prediction:", classname[pred])


if __name__ == '__main__':
    main()
