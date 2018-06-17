from keras.models import Model
from keras.layers import *
from keras.layers.core import Activation, Layer

'''
Keras Customizable Residual Unit
This is a simplified implementation of the basic (no bottlenecks) full pre-activation residual unit from He, K., Zhang, X., Ren, S., Sun, J., "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027v2).
'''

def conv_block(feat_maps_out, prev):
    prev = BatchNormalization()(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv3D(feat_maps_out, (3, 3, 5), padding="same")(prev) 
    prev = BatchNormalization()(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv3D(feat_maps_out, (3, 3, 5), padding="same")(prev)   
 
    return prev


def identitiy_fix_size(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Conv3D(feat_maps_out, (1, 1, 1), padding='same')(prev)
    return prev 


def Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    id = identitiy_fix_size(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
    return merge([id, conv], mode='sum') # the residual connection


def dense_block(feat_maps_out, prev):
    prev = Dense(feat_maps_out, activation='relu', kernel_initializer='he_normal')(prev)
    prev = Dropout(rate=0.4)(prev)
    prev = BatchNormalization()(prev)
    prev = Dense(feat_maps_out, activation='relu', kernel_initializer='he_normal')(prev)
    prev = Dropout(rate=0.4)(prev)
    prev = BatchNormalization()(prev)
    return prev
    
def identitiy_fix_size_dense(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Dense(feat_maps_out, activation='relu', kernel_initializer='he_normal')(prev)
    return prev


def Dense_Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A residual unit with dense blocks 
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    id = identitiy_fix_size_dense(feat_maps_in, feat_maps_out, prev_layer)
    dense = dense_block(feat_maps_out, prev_layer)

    return merge([id, dense], mode='sum') # the residual connection





















