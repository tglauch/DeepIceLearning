import keras.backend as K
import numpy as np
import tensorflow as tf

def azimuth_mse(yTrue,yPred):
    return K.mean(K.square(tf.atan2(tf.sin(yTrue - yPred), tf.cos(yTrue - yPred))))

 
def zenith_mse(yTrue,yPred):
    return K.mean(K.square(yTrue - yPred)) 

def cart_out_loss(yTrue, yPred):
    return K.mean(K.square(yTrue - yPred))
