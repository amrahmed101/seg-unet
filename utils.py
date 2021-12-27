import os
import imageio
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

labels = pd.read_csv('labelsunet.csv',index_col=0)
mapping_dict = {label:list(labels.loc[label,:]) for label in labels.index}
def preprocess_mask(classes,mask_img):
    ''' 
    preprocess the mask image to return OHE mask
    '''
    stack_list = []
    # 1- get the image(skipped for now)
    # 2- look for pixels = class pixels
    for cl in classes.values():
        equals = tf.equal(mask_img,cl)
    # 3- OHE those pixels in an array
        f = np.all(equals,axis=-1)
        ohe = tf.cast(f,dtype=tf.int32)
        stack_list.append(ohe)
    # 4- convert the array to a tensor
    annotations = tf.stack(stack_list,axis = -1)
    return annotations


reverse_mask = {}

for idx ,(key,val) in enumerate(mapping_dict.items()):
    reverse_mask[idx] = np.array(val)   
idx2rgb={idx:np.array(rgb) for idx, (cl, rgb) in enumerate(mapping_dict.items())}


def map_class_to_rgb(p):
  
  return idx2rgb[p[0]]


def predict_visualize(image,alpha = 0.7,mode=None):
    '''
    Predicts image mask and make overlayed mask for visaualization
    inputs :
        image_path : path for image to be predicted
        alpha : alpha value for mask overlay
        
    returns :
        pred : predicted mask image of shape (256,256,32)
        
    '''
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(256,256),interpolation = cv2.INTER_AREA)
    
    pred = mode.predict(np.expand_dims(image,0)/255)
    
    pred = np.apply_along_axis(map_class_to_rgb, -1, np.expand_dims(np.argmax(pred, axis=-1), -1))
    pred_vis = np.reshape(pred,(256,256,3))

    vis = cv2.addWeighted(image,1.,pred_vis,alpha,0, dtype = cv2.CV_32F)/255

    return vis