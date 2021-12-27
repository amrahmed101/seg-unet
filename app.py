
from unet import *
from utils import *
import numpy as np 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from flask import Flask,render_template,Response

app=Flask(__name__)

model =build_unet((256,256,3),stage_filters=[64,128,256,512],n_classes=32)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4) ,loss = 'categorical_crossentropy',metrics= ['accuracy'])
model.load_weights('unet_20_20epochs.hdf5')

camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            pred = predict_visualize(frame,alpha = 0.8,mode=model)
            pred=(pred * 255).astype(np.uint8)
            ret,buffer=cv2.imencode('.jpg',pred)
            pred=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + pred + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run()