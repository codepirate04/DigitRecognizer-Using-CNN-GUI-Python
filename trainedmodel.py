from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
model=load_model('digitreg.h5')

def classify():
    image= cv.imread('image.png',cv.IMREAD_GRAYSCALE)
    image=cv.resize(image,(28,28))
    image=image.reshape(1,28,28,1)
    image=image/255
    prediction=model.predict(image)
    store=np.copy(prediction)
    store[0,store.argmax()]=0
    max2=store.argmax()
    print("The  predicted number is:",prediction.argmax(),", Second Guess:",max2)
    
    
