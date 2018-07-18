"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
from keras import backend as K
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

"""
def event_type_and_energy_weighted_loss(deposited_energy_event, classification_event):

   # Annahme. Wir haben die histogramme und Uebergabe klappt
   deposited_energy_event = np.log10(deposited_energy_event)
   if classification_event == 1.0:
        N_bin = vals_1[np.digitize(deposited_energy_event, bins_1)]
        weights = np.array([1./N_bins, 1, 1])
   elif classification_event == 2.0:
       N_bin = vals_2[np.digitize(deposited_energy_event, bins_2)]
       weights = np.array([1, 1./N_bins, 1])
   else: 
       N_bin = vals_3[np.digitize(deposited_energy_event, bins_3)] 
       weights = np.array([1, 1, 1./N_bins])   

   weights = K.variable(weights)

   def loss(y_true, y_pred):
       # scale predictions so that the class probas of each sample sum to 1
       y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
       # clip to prevent NaN's and Inf's
       y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
       # calc
       loss = y_true * K.log(y_pred) * weights
       loss = -K.sum(loss, -1)
       return loss
   return loss

"""



