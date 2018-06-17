#!/usr/bin/env python
# coding: utf-8

import keras
from keras.layers import *


def residual_unit_2DConv (x0, size):
    ''' input: one tensor of shape (n , nx , ny , size)'''
    x = Convolution2D (size , (1 , 1) , border_mode = " same " , activation = ’ relu ’) (x0)
    x = Convolution2D (size , (3 , 3) , border_mode = " same " , activation = ’ relu ’) (x)
    return add ([x ,x0])

def inception_unit (x0, size):
    x1 = Convolution2D (16 , (1 , 1) , padding = ’ same ’ , activation = ’ relu ’) ( x0 )
    x1 = Convolution2D (16 , (3 , 3) , padding = ’ same ’ , activation = ’ relu ’) ( x1 )
    x2 = Convolution2D (16 , (1 , 1) , padding = ’ same ’ , activation = ’ relu ’) ( x0 )
    x2 = Convolution2D (16 , (5 , 5) , padding = ’ same ’ , activation = ’ relu ’) ( x2 )
    x3 = Convolution2D (16 , (1 , 1) , padding = ’ same ’ , activation = ’ relu ’) ( x0 )
    x4 = MaxPooling2D ((3 , 3) , strides =(1 , 1) , padding = ’ same ’) ( x0 )
    x4 = Convolution2D (64 , (1 , 1) , padding = ’ same ’ , activation = ’ relu ’) ( x4 )
    return concatenate ([ x1 , x2 , x3 , x4 ] , axis =1)

