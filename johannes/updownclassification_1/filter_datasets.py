
# coding: utf-8

# In[1]:


import numpy as np
import tables


pulses_file = "/data/user/jkager/files/inicedstpulses_nugen11069_first50i3files.h5"

# In[4]:


h5=tables.open_file(pulses_file)
mcp=h5.root.MCPrimary.cols
data =   np.load("updownclassification_using_keras_chargedata.npy")
labels = np.load("updownclassification_using_keras_labels.npy")


# In[5]:


data_filtered = []
labels_filtered = []
included_zeniths = []
for i, entry in enumerate(h5.root.MCPrimary.where("((Run != 1106900050) | (Event != 1612) | (SubEvent != 1))")):
    if entry['zenith'] > 7*np.pi/12 or entry['zenith'] < 5*np.pi/12:
        data_filtered.append(data[i])
        labels_filtered.append(labels[i])
        included_zeniths.append(entry['zenith'])
data_filtered = np.array(data_filtered)
labels_filtered = np.array(labels_filtered)
included_zeniths = np.array(included_zeniths)


# In[6]:

np.save("updownclassification_using_keras_zenith_filtered.npy", included_zeniths)
np.save("updownclassification_using_keras_chargedata_filtered.npy", data_filtered)
np.save("updownclassification_using_keras_labels_filtered.npy", labels_filtered)


# filtering done
# -



